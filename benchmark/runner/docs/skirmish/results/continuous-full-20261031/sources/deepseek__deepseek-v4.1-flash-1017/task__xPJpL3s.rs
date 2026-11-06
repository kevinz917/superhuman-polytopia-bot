// Bardur Polytopia 1v1 bot: heuristic economy + military agent.
use serde_json::Value;
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy)]
struct Un {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f64,
    moved: bool,
    attacked: bool,
    veteran: bool,
    kills: i32,
    home: i32,
}

#[derive(Clone, Copy)]
struct Ci {
    tile: usize,
    owner: i32,
    level: i32,
    production: i32,
    rewards: [i32; 9],
}

struct Obs {
    size: i32,
    n: usize,
    known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    road: Vec<bool>,
    capital: Vec<i32>,
    units: Vec<Un>,
    unit_at: Vec<i32>,
    cities: Vec<Ci>,
    stars: i32,
    tech: Vec<bool>,
    turn: i32,
    me: i32,
}

#[inline]
fn cheb(size: i32, a: usize, b: usize) -> i32 {
    let (ax, ay) = ((a as i32) % size, (a as i32) / size);
    let (bx, by) = ((b as i32) % size, (b as i32) / size);
    (ax - bx).abs().max((ay - by).abs())
}

fn parse(v: &Value) -> Obs {
    let vs = &v["visible_state"];
    let size = vs["board_size"].as_i64().unwrap_or(14) as i32;
    let n = (size * size) as usize;
    let mut o = Obs {
        size,
        n,
        known: vec![false; n],
        terrain: vec![0; n],
        owner: vec![0; n],
        structure: vec![0; n],
        resource: vec![0; n],
        road: vec![false; n],
        capital: vec![0; n],
        units: Vec::new(),
        unit_at: vec![-1; n],
        cities: Vec::new(),
        stars: vs["stars"].as_i64().unwrap_or(0) as i32,
        tech: vec![false; 128],
        turn: vs["turn"].as_i64().unwrap_or(0) as i32,
        me: v["player"].as_i64().unwrap_or(1) as i32,
    };
    if let Some(ts) = vs["tiles"].as_array() {
        for t in ts {
            let i = t["tile"].as_i64().unwrap_or(-1);
            if i < 0 || i as usize >= n {
                continue;
            }
            let i = i as usize;
            if t["known"].as_bool().unwrap_or(false) {
                o.known[i] = true;
                o.terrain[i] = t["terrain"].as_i64().unwrap_or(0) as i32;
                o.owner[i] = t["owner"].as_i64().unwrap_or(0) as i32;
                o.structure[i] = t["structure"].as_i64().unwrap_or(0) as i32;
                o.resource[i] = t["resource"].as_i64().unwrap_or(0) as i32;
                o.road[i] = t["road"].as_i64().unwrap_or(0) != 0;
                o.capital[i] = t["capital"].as_i64().unwrap_or(0) as i32;
            }
        }
    }
    if let Some(ts) = vs["technology"].as_array() {
        for t in ts {
            if let Some(k) = t.as_i64() {
                if (0..128).contains(&k) {
                    o.tech[k as usize] = true;
                }
            }
        }
    }
    if let Some(us) = vs["units"].as_array() {
        for u in us {
            let tile = u["tile"].as_i64().unwrap_or(-1);
            if tile < 0 || tile as usize >= n {
                continue;
            }
            let tile = tile as usize;
            let un = Un {
                tile,
                owner: u["owner"].as_i64().unwrap_or(0) as i32,
                kind: u["kind"].as_i64().unwrap_or(0) as i32,
                health: u["health"].as_f64().unwrap_or(10.0),
                moved: u["moved"].as_bool().unwrap_or(false),
                attacked: u["attacked"].as_bool().unwrap_or(false),
                veteran: u["veteran"].as_bool().unwrap_or(false),
                kills: u["kills"].as_i64().unwrap_or(0) as i32,
                home: u["home"].as_i64().unwrap_or(0) as i32,
            };
            o.unit_at[tile] = o.units.len() as i32;
            o.units.push(un);
        }
    }
    if let Some(cs) = vs["cities"].as_array() {
        for c in cs {
            let tile = c["tile"].as_i64().unwrap_or(-1);
            if tile < 0 || tile as usize >= n {
                continue;
            }
            let mut rw = [0i32; 9];
            if let Some(rs) = c["rewards"].as_array() {
                for k in 0..9 {
                    rw[k] = rs.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32;
                }
            }
            o.cities.push(Ci {
                tile: tile as usize,
                owner: c["owner"].as_i64().unwrap_or(0) as i32,
                level: c["level"].as_i64().unwrap_or(1) as i32,
                production: c["production"].as_i64().unwrap_or(1) as i32,
                rewards: rw,
            });
        }
    }
    o
}

fn stats(kind: i32) -> (f64, f64, f64, i32, i32) {
    // hp, attack, defense, range, cost
    match kind {
        2 => (10.0, 2.0, 2.0, 1, 2),
        3 => (10.0, 2.0, 1.0, 1, 3),
        4 => (10.0, 3.5, 1.0, 1, 8),
        5 => (15.0, 1.0, 3.0, 1, 3),
        8 => (10.0, 4.0, 0.0, 3, 8),
        9 => (10.0, 2.0, 1.0, 2, 3),
        10 => (10.0, 0.0, 1.0, 1, 5),
        11 => (15.0, 3.0, 3.0, 1, 5),
        12 => (40.0, 5.0, 4.0, 1, 10),
        15 => (15.0, 3.0, 1.0, 1, 2),
        37 => (10.0, 2.0, 2.0, 1, 2),
        38 => (5.0, 0.0, 0.5, 1, 8),
        39 => (5.0, 0.0, 0.5, 1, 2),
        40 => (10.0, 2.0, 2.0, 1, 2),
        41 => (10.0, 3.0, 2.0, 3, 15),
        42 => (10.0, 2.0, 1.0, 2, 5),
        43 => (10.0, 0.0, 1.0, 2, 0),
        44 => (10.0, 3.0, 3.0, 1, 5),
        45 => (40.0, 4.0, 4.0, 1, 10),
        _ => (10.0, 2.0, 2.0, 1, 3),
    }
}

fn is_stiff(kind: i32) -> bool {
    matches!(kind, 8 | 10 | 17 | 22 | 25 | 38 | 39 | 41 | 43 | 45 | 60 | 61)
}

fn fortifies(kind: i32) -> bool {
    matches!(kind, 2 | 3 | 5 | 9 | 15 | 20 | 25 | 37)
}

fn combat_est(
    aatk: f64,
    ahp: f64,
    amax: f64,
    ddef: f64,
    dhp: f64,
    dmax: f64,
    bonus: f64,
    drange: i32,
    dist: i32,
    dstiff: bool,
) -> (f64, f64) {
    let af = aatk * (ahp / amax.max(1.0));
    let df = ddef * (dhp / dmax.max(1.0)) * bonus;
    let total = af + df;
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    let damage = ((af / total) * aatk * 4.5 + 1e-6 + 0.5).floor();
    let retal = if dhp > damage && drange >= dist && !dstiff {
        ((df / total) * ddef * 4.5 + 1e-6 + 0.5).floor()
    } else {
        0.0
    };
    (damage, retal)
}

#[derive(Clone, Copy)]
struct Cmd {
    kind: i32,
    src: i32,
    dst: i32,
    param: i32,
    idx: usize,
}

struct Plan {
    my_cities: Vec<usize>,
    villages: Vec<usize>,
    ruins: Vec<usize>,
    unexplored: Vec<usize>,
    enemy_units: Vec<usize>,
    enemy_cities: Vec<usize>,
    city_threat: Vec<(usize, i32)>,
    goals: Vec<Option<usize>>,
    army: usize,
    cities: usize,
    attack_mode: bool,
    want_unit: bool,
    front: Option<usize>,
    threat: Vec<i32>,
}

fn build_plan(o: &Obs) -> Plan {
    let mut villages = Vec::new();
    let mut ruins = Vec::new();
    let mut unexplored = Vec::new();
    let mut enemy_units = Vec::new();
    let mut enemy_cities = Vec::new();
    let mut my_cities = Vec::new();
    for i in 0..o.n {
        if !o.known[i] {
            unexplored.push(i);
            continue;
        }
        match o.structure[i] {
            1 => {
                if o.owner[i] == 0 {
                    villages.push(i);
                } else if o.owner[i] != o.me {
                    enemy_cities.push(i);
                }
            }
            2 => {
                if o.owner[i] == 0 {
                    ruins.push(i);
                }
            }
            _ => {}
        }
        if o.owner[i] == o.me && o.structure[i] == 1 {
            my_cities.push(i);
        }
    }
    let mut army = 0usize;
    for u in &o.units {
        if u.owner == o.me {
            army += 1;
        } else {
            enemy_units.push(u.tile);
        }
    }
    if my_cities.is_empty() {
        for c in &o.cities {
            if c.owner == o.me {
                my_cities.push(c.tile);
            }
        }
    }
    let mut city_threat = Vec::new();
    for &c in &my_cities {
        let mut best = 99;
        for &e in &enemy_units {
            let d = cheb(o.size, c, e);
            if d < best {
                best = d;
            }
        }
        city_threat.push((c, best));
    }
    let threats = city_threat.iter().filter(|(_, d)| *d <= 7).count();
    let target = (5 + 2 * my_cities.len() + threats).min(26);
    let attack_mode = o.turn >= 120;
    let base0 = my_cities.first().copied().unwrap_or(0);
    let mut front = None;
    let mut bestd = 999;
    for &e in &enemy_units {
        let d = cheb(o.size, base0, e);
        if d < bestd {
            bestd = d;
            front = Some(e);
        }
    }
    for &e in &enemy_cities {
        let d = cheb(o.size, base0, e);
        if d < bestd {
            bestd = d;
            front = Some(e);
        }
    }
    let mut p = Plan {
        my_cities,
        villages,
        ruins,
        unexplored,
        enemy_units,
        enemy_cities,
        city_threat,
        goals: Vec::new(),
        army,
        cities: 0,
        attack_mode,
        want_unit: army < target,
        front,
        threat: vec![0; o.n],
    };
    p.cities = o.cities.iter().filter(|c| c.owner == o.me).count();
    for u in &o.units {
        if u.owner == o.me {
            continue;
        }
        let (_, atk, _, range, _) = stats(u.kind);
        if atk <= 0.0 {
            continue;
        }
        let reach = 2 + range;
        let (ex, ey) = ((u.tile as i32) % o.size, (u.tile as i32) / o.size);
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let nx = ex + dx;
                let ny = ey + dy;
                if nx < 0 || ny < 0 || nx >= o.size || ny >= o.size {
                    continue;
                }
                let t = (ny * o.size + nx) as usize;
                p.threat[t] += 1;
            }
        }
    }
    // ---- goals: fight nearby enemies, otherwise expand -------------------
    let mut goals: Vec<Option<usize>> = vec![None; o.units.len()];
    for ui in 0..o.units.len() {
        let u = o.units[ui];
        if u.owner != o.me {
            continue;
        }
        let src = u.tile;
        let mut best = f64::NEG_INFINITY;
        let mut bt = None;
        for &e in &p.enemy_units {
            let d = cheb(o.size, src, e);
            if d <= 4 {
                let v = 460.0 - 25.0 * d as f64;
                if v > best {
                    best = v;
                    bt = Some(e);
                }
            }
        }
        for &v in &p.villages {
            let d = cheb(o.size, src, v);
            let sc = 420.0 - 9.0 * d as f64;
            if sc > best {
                best = sc;
                bt = Some(v);
            }
        }
        for &v in &p.ruins {
            let d = cheb(o.size, src, v);
            let sc = 380.0 - 9.0 * d as f64;
            if sc > best {
                best = sc;
                bt = Some(v);
            }
        }
        if bt.is_none() || best < 0.0 {
            let mut bd = 999;
            let mut bu = None;
            for &t in &p.unexplored {
                let d = cheb(o.size, src, t);
                if d < bd {
                    bd = d;
                    bu = Some(t);
                }
            }
            if let Some(t) = bu {
                if best < 200.0 {
                    bt = Some(t);
                }
            }
        }
        if bt.is_none() {
            let mut bd = 999;
            for &e in &p.enemy_units {
                let d = cheb(o.size, src, e);
                if d < bd {
                    bd = d;
                    bt = Some(e);
                }
            }
            for &e in &p.enemy_cities {
                let d = cheb(o.size, src, e);
                if d < bd {
                    bd = d;
                    bt = Some(e);
                }
            }
        }
        goals[ui] = bt;
    }
    // wounded units fall back to the closest own city
    for ui in 0..o.units.len() {
        let u = o.units[ui];
        if u.owner != o.me {
            continue;
        }
        let (hp, _, _, _, _) = stats(u.kind);
        let amax = hp + if u.veteran { 5.0 } else { 0.0 };
        if u.health < amax * 0.5 {
            let mut near = false;
            for &e in &p.enemy_units {
                if cheb(o.size, u.tile, e) <= 4 {
                    near = true;
                }
            }
            if near {
                let mut bd = 999;
                let mut bc = None;
                for &c in &p.my_cities {
                    let d = cheb(o.size, u.tile, c);
                    if d < bd {
                        bd = d;
                        bc = Some(c);
                    }
                }
                if let Some(c) = bc {
                    goals[ui] = Some(c);
                }
            }
        }
    }
    p.goals = goals;
    p
}

fn reveal_count(o: &Obs, dst: usize) -> i32 {
    let mut c = 0;
    let (x, y) = ((dst as i32) % o.size, (dst as i32) / o.size);
    for dy in -2..=2i32 {
        for dx in -2..=2i32 {
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || ny < 0 || nx >= o.size || ny >= o.size {
                continue;
            }
            let t = (ny * o.size + nx) as usize;
            if !o.known[t] {
                c += 1;
            }
        }
    }
    c
}

fn tech_priority(o: &Obs) -> Vec<i32> {
    // counts of useful tiles we own
    let mut forests = 0;
    let mut fruit = 0;
    let mut metal = 0;
    let mut water = 0;
    for i in 0..o.n {
        if !o.known[i] || o.owner[i] != o.me || o.structure[i] != 0 {
            continue;
        }
        match o.terrain[i] {
            5 => forests += 1,
            4 => {
                if o.resource[i] == 5 {
                    metal += 1
                }
            }
            3 => {
                if o.resource[i] == 6 {
                    fruit += 1
                }
            }
            1 | 2 => water += 1,
            _ => {}
        }
    }
    let has = |k: i32| -> bool { k >= 0 && k < 128 && o.tech[k as usize] };
    let mut out: Vec<i32> = Vec::new();
    if !has(18) {
        out.push(18); // archery
    }
    if !has(16) && forests >= 2 {
        out.push(16); // forestry
    }
    if !has(6) && fruit >= 1 {
        out.push(6); // organization (fruit)
    }
    if !has(20) && metal >= 1 {
        out.push(20);
    }
    if !has(23) && metal >= 1 {
        out.push(23); // mining
    }
    if !has(10) && water >= 6 {
        out.push(10); // fishing
    }
    if !has(17) && has(16) && forests >= 2 {
        out.push(17); // mathematics
    }
    if !has(7) && has(6) {
        out.push(7);
    }
    if !has(24) && has(23) {
        out.push(24); // smithery
    }
    if !has(1) {
        out.push(1);
    }
    if !has(4) && has(1) {
        out.push(4);
    }
    out
}

fn research_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    let pri = tech_priority(o);
    let mut found = None;
    for (i, &k) in pri.iter().enumerate() {
        if k == c.param {
            found = Some(i);
            break;
        }
    }
    let idx = match found {
        Some(i) => i,
        None => return -1.0,
    };
    let tier = match c.param {
        0 | 1 | 6 | 10 | 15 | 20 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 | 121 => 2,
        _ => 3,
    };
    let cost: i32 = 4 + (p.cities.max(1) as i32) * tier;
    let reserve = if p.want_unit { 7 } else { 0 };
    let reserve = if idx <= 1 { reserve } else { reserve + 8 };
    if o.stars - cost < reserve {
        return -1.0;
    }
    if idx >= 3 {
        1200.0 - idx as f64 * 50.0
    } else {
        3500.0 - idx as f64 * 80.0
    }
}

fn capture_bonus(o: &Obs, c: &Cmd) -> f64 {
    let t = c.src as usize;
    if t >= o.n {
        return 0.0;
    }
    if o.structure[t] == 1 {
        if o.owner[t] != 0 && o.owner[t] != o.me {
            if o.capital[t] != 0 {
                return 3000.0;
            }
            return 1500.0;
        }
        return 900.0;
    }
    if o.structure[t] == 2 {
        return 700.0;
    }
    400.0
}

fn attack_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    let src = c.src as usize;
    let dst = c.dst as usize;
    let ai = o.unit_at[src];
    let di = o.unit_at[dst];
    if ai < 0 || di < 0 {
        return -1.0;
    }
    let a = &o.units[ai as usize];
    let d = &o.units[di as usize];
    if a.owner != o.me || d.owner == o.me {
        return -1.0;
    }
    let (ahp0, aatk, _, arange, acost) = stats(a.kind);
    let (dhp0, _, ddef, drange, dcost) = stats(d.kind);
    let amax = ahp0 + if a.veteran { 5.0 } else { 0.0 };
    let dmax = dhp0 + if d.veteran { 5.0 } else { 0.0 };
    let dist = cheb(o.size, src, dst);
    let in_city = o.structure[dst] == 1 && o.owner[dst] == d.owner;
    let bonus = if in_city && fortifies(d.kind) { 1.5 } else { 1.0 };
    let (dmg, retal) = combat_est(aatk, a.health, amax, ddef, d.health, dmax, bonus, drange, dist, is_stiff(d.kind));
    let kill = dmg >= d.health;
    let mut s = 2.2 * dmg - 2.4 * retal;
    if kill {
        s += 30.0 + 5.0 * dcost as f64;
        if in_city {
            s += 120.0;
        }
    }
    // urgency: enemy next to one of our cities
    let mut urgent = false;
    for &(cc, _) in &p.city_threat {
        if cheb(o.size, dst, cc) <= 1 {
            urgent = true;
        }
    }
    if urgent {
        s += 25.0;
    }
    if dmg <= 0.0 {
        return -1.0;
    }
    if !kill && retal >= dmg && !urgent {
        return -1.0;
    }
    let _ = arange;
    let _ = acost;
    if s <= 14.0 {
        return -1.0;
    }
    5000.0 + s
}

fn ability_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    match c.param {
        16 => 6000.0,
        7 => {
            // recover: heal in place
            let src = c.src as usize;
            let ai = o.unit_at[src];
            if ai < 0 {
                return -1.0;
            }
            let a = &o.units[ai as usize];
            let (hp, _, _, _, _) = stats(a.kind);
            let amax = hp + if a.veteran { 5.0 } else { 0.0 };
            if a.health <= amax * 0.45 {
                let mut near = false;
                for &e in &p.enemy_units {
                    if cheb(o.size, src, e) <= 2 {
                        near = true;
                    }
                }
                if near {
                    return 4000.0 * (1.0 - a.health / amax);
                }
            }
            -1.0
        }
        _ => -1.0,
    }
}

fn harvest_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    let _ = (p, c);
    let _ = o;
    3000.0
}

fn build_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    let t = c.dst as usize;
    if t >= o.n {
        return -1.0;
    }
    match c.param {
        23..=29 => 5600.0, // free monument: +3 population
        12 => 2600.0,      // lumber hut
        21 => 2700.0,      // mine
        13 => {
            let mut adj = 0;
            let (x, y) = ((t as i32) % o.size, (t as i32) / o.size);
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= o.size || ny >= o.size {
                        continue;
                    }
                    let j = (ny * o.size + nx) as usize;
                    if o.structure[j] == 12 {
                        adj += 1;
                    }
                }
            }
            if adj == 0 {
                return -1.0;
            }
            2500.0 + 400.0 * adj as f64
        }
        8 => {
            if o.tech[10] {
                1500.0
            } else {
                -1.0
            }
        }
        50 => 2400.0,
        3 => {
            if o.stars >= 14 {
                450.0
            } else {
                -1.0
            }
        }
        _ => -1.0,
    }
}

fn summon_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    let t = c.src as usize;
    if t >= o.n {
        return -1.0;
    }
    let kind = c.param;
    let (_, _, _, _, cost) = stats(kind);
    if o.stars < cost {
        return -1.0;
    }
    let mut pref = match kind {
        3 => 48.0,
        9 => 44.0,
        2 => 40.0,
        11 => 38.0,
        5 => 20.0,
        8 => 10.0,
        _ => -1.0,
    };
    // hold the line with warriors right next to the enemy
    if kind == 2 {
        let mut near = false;
        for &e in &p.enemy_units {
            if cheb(o.size, t, e) <= 5 {
                near = true;
            }
        }
        if near {
            pref += 18.0;
        }
    }
    if pref < 0.0 {
        return -1.0;
    }
    // melee / ranged balance
    let mut melee = 0;
    let mut ranged = 0;
    for u in &o.units {
        if u.owner != o.me {
            continue;
        }
        let (_, _, _, r, _) = stats(u.kind);
        if r >= 2 {
            ranged += 1;
        } else {
            melee += 1;
        }
    }
    if kind == 9 && melee < ranged + 1 {
        pref -= 25.0;
    }
    if kind == 2 && ranged + 1 < melee {
        pref -= 15.0;
    }
    // prefer recruiting near the frontier
    let mut d = 0.0;
    if let Some(f) = p.front {
        d = cheb(o.size, t, f) as f64;
    }
    let base = if p.want_unit { 2750.0 } else { 1150.0 };
    base + pref - d * 6.0
}

fn move_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    let src = c.src as usize;
    let dst = c.dst as usize;
    if src >= o.n || dst >= o.n || src == dst {
        return -1.0;
    }
    let ai = o.unit_at[src];
    if ai < 0 {
        return -1.0;
    }
    let ui = ai as usize;
    let u = o.units[ui];
    if u.owner != o.me {
        return -1.0;
    }
    let goal = match p.goals.get(ui).copied().flatten() {
        Some(g) => g,
        None => return -1.0,
    };
    let d0 = cheb(o.size, src, goal);
    let d1 = cheb(o.size, dst, goal);
    let gain = d0 - d1;
    let rev = reveal_count(o, dst);
    if gain <= 0 && rev < 5 {
        return -1.0;
    }
    let (uhp, _, _, range, _) = stats(u.kind);
    let umax = uhp + if u.veteran { 5.0 } else { 0.0 };
    let mut s = 700.0 + 16.0 * gain as f64 + rev as f64 * 2.0;
    let in_city = o.structure[dst] == 1 && o.owner[dst] == o.me;
    let mut attackable = 0;
    for &e in &p.enemy_units {
        if cheb(o.size, dst, e) <= range {
            attackable += 1;
        }
    }
    if attackable > 0 {
        s += 70.0;
    }
    let th = p.threat.get(dst).copied().unwrap_or(0) as f64;
    if in_city {
        s += 220.0 - 10.0 * th;
    } else if o.owner[dst] == o.me {
        s += 60.0 - 45.0 * th;
    } else {
        s -= 70.0 * th;
    }
    if u.health < umax * 0.6 {
        s -= 60.0 * th;
    }
    if o.structure[dst] == 1 && o.owner[dst] == 0 {
        s += 320.0;
    }
    if o.structure[src] == 1 && o.owner[src] == o.me && o.stars >= 2 {
        let mut enemy_close = false;
        for &e in &p.enemy_units {
            if cheb(o.size, src, e) <= 3 {
                enemy_close = true;
            }
        }
        if !enemy_close {
            s += 380.0;
        }
    }
    if o.structure[dst] == 2 {
        s += 120.0;
    }
    if o.terrain[dst] == 5 || o.terrain[dst] == 4 {
        s += 4.0;
    }
    s
}

fn reward_score(o: &Obs, c: &Cmd) -> f64 {
    let unknown = o.known.iter().filter(|k| !**k).count() as f64 / o.n.max(1) as f64;
    match c.param {
        8 => 100.0, // population
        3 => 95.0,  // workshop
        6 => 92.0,  // giant
        4 => {
            if o.turn < 20 && unknown > 0.45 {
                96.0
            } else {
                55.0
            }
        } // explorer
        2 => 70.0,  // park
        7 => 68.0,  // +5 stars
        5 => 50.0,  // border growth
        1 => {
            // walls: only offered at level 3. Take them for cities under pressure.
            let ct = c.dst as usize;
            let mut near = 99;
            for &e in &o.units {
                if e.owner != o.me {
                    let d = cheb(o.size, ct, e.tile);
                    if d < near {
                        near = d;
                    }
                }
            }
            if near <= 9 {
                99.0
            } else {
                45.0
            }
        }
        _ => 10.0,
    }
}

fn action_score(o: &Obs, p: &Plan, c: &Cmd) -> f64 {
    match c.kind {
        1 => move_score(o, p, c),
        2 => attack_score(o, p, c),
        3 => ability_score(o, p, c),
        4 => summon_score(o, p, c),
        5 => harvest_score(o, p, c),
        6 => build_score(o, p, c),
        7 => research_score(o, p, c),
        8 => 10000.0 + capture_bonus(o, c),
        _ => -1.0,
    }
}

fn choose(obs: &Value) -> String {
    let o = parse(obs);
    let legal = match obs["legal_actions"].as_array() {
        Some(a) => a,
        None => return String::new(),
    };
    let mut cmds: Vec<Cmd> = Vec::with_capacity(legal.len());
    let mut endturn: Option<usize> = None;
    for (i, a) in legal.iter().enumerate() {
        let c = &a["command"];
        let kind = c["kind"].as_i64().unwrap_or(10) as i32;
        if kind == 10 && endturn.is_none() {
            endturn = Some(i);
        }
        cmds.push(Cmd {
            kind,
            src: c["source"].as_i64().unwrap_or(-1) as i32,
            dst: c["target"].as_i64().unwrap_or(-1) as i32,
            param: c["parameter"].as_i64().unwrap_or(0) as i32,
            idx: i,
        });
    }
    // mandatory city rewards
    let mut has_reward = false;
    let mut best_reward: Option<&Cmd> = None;
    let mut bs_r = f64::NEG_INFINITY;
    for c in &cmds {
        if c.kind == 9 {
            has_reward = true;
            let s = reward_score(&o, c);
            if s > bs_r {
                bs_r = s;
                best_reward = Some(c);
            }
        }
    }
    if has_reward {
        if let Some(c) = best_reward {
            return legal[c.idx]["action_id"].as_str().unwrap_or("").to_owned();
        }
    }
    let p = build_plan(&o);
    let mut best: Option<usize> = None;
    let mut bs = 0.0f64;
    for c in &cmds {
        let s = action_score(&o, &p, c);
        if s > bs {
            bs = s;
            best = Some(c.idx);
        }
    }
    let idx = match best {
        Some(i) => i,
        None => match endturn {
            Some(i) => i,
            None => 0,
        },
    };
    legal[idx]["action_id"].as_str().unwrap_or("").to_owned()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let obs: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = choose(&obs);
        if id.is_empty() {
            continue;
        }
        let out = serde_json::json!({"action_id": id});
        if writeln!(stdout, "{}", out).is_err() {
            break;
        }
        if stdout.flush().is_err() {
            break;
        }
    }
}
