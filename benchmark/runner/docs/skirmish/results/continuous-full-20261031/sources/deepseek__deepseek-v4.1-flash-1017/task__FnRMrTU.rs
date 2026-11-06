// Polytopia 1v1 Bardur bot: heuristic scorer over the legal action catalog.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy, Default)]
struct Unit {
    owner: i32,
    kind: i32,
    health: f32,
    maxhp: f32,
    tile: i32,
    attacked: bool,
    moved: bool,
    kills: i32,
    veteran: bool,
}

#[derive(Clone, Copy, Default)]
struct City {
    tile: i32,
    owner: i32,
    level: i32,
    production: i32,
    progress: i32,
}

#[derive(Clone)]
struct Act {
    id: String,
    kind: i32,
    source: i32,
    target: i32,
    param: i32,
}

struct W {
    size: i32,
    n: i32,
    player: i32,
    turn: i32,
    stars: i32,
    known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    road: Vec<bool>,
    capital: Vec<i32>,
    units: Vec<Unit>,
    own: Vec<City>,
    enemy_cities: Vec<City>,
    neutral_villages: Vec<i32>,
    tech: [bool; 128],
    peace: bool,
    met: bool,
    actions: Vec<Act>,
    enemy_near_city: i32,
    unit_targets: Vec<(i32, i32)>,
    city_threat: Vec<(i32, i32)>,
    my_units: i32,
    enemy_units: i32,
    unknown: i32,
    forest_own: i32,
    crop_own: i32,
    metal_own: i32,
    fruit_own: i32,
}

#[inline]
fn stat(kind: i32) -> (f32, f32, f32, i32, i32, i32) {
    // hp, attack, defense, range, movement, cost
    match kind {
        2 => (10.0, 2.0, 2.0, 1, 1, 2),
        3 => (10.0, 2.0, 1.0, 1, 2, 3),
        4 => (10.0, 3.5, 1.0, 1, 3, 8),
        5 => (15.0, 1.0, 3.0, 1, 1, 3),
        8 => (10.0, 4.0, 0.0, 3, 1, 8),
        9 => (10.0, 2.0, 1.0, 2, 1, 3),
        10 => (10.0, 0.0, 1.0, 1, 1, 5),
        11 => (15.0, 3.0, 3.0, 1, 1, 5),
        12 => (40.0, 5.0, 4.0, 1, 1, 10),
        20 => (10.0, 2.0, 1.0, 1, 2, 3),
        21 => (10.0, 2.5, 1.0, 2, 2, 8),
        37 => (10.0, 2.0, 2.0, 1, 1, 2),
        38 => (5.0, 0.0, 0.5, 1, 2, 8),
        39 => (5.0, 0.0, 0.5, 1, 2, 2),
        40 => (10.0, 2.0, 2.0, 1, 2, 2),
        41 => (10.0, 3.0, 2.0, 3, 2, 15),
        42 => (10.0, 2.0, 1.0, 2, 3, 5),
        43 => (10.0, 0.0, 1.0, 2, 2, 0),
        44 => (10.0, 3.0, 3.0, 1, 3, 5),
        45 => (40.0, 4.0, 4.0, 1, 2, 10),
        56 => (10.0, 3.0, 3.0, 1, 2, 5),
        _ => (10.0, 2.0, 1.0, 1, 1, 3),
    }
}

#[inline]
fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let div = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / div) * aa * 4.5 } else { 0.0 }) + 0.000001 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / div) * dd * 4.5 } else { 0.0 }) + 0.000001 + 0.5).floor();
    (attack, defense)
}

impl W {
    fn parse(o: &Value) -> W {
        let vs = &o["visible_state"];
        let size = vs["board_size"].as_i64().unwrap_or(14) as i32;
        let n = size * size;
        let player = o["player"].as_i64().unwrap_or(1) as i32;
        let turn = vs["turn"].as_i64().unwrap_or(0) as i32;
        let stars = vs["stars"].as_i64().unwrap_or(0) as i32;
        let ni = n as usize;
        let mut w = W {
            size,
            n,
            player,
            turn,
            stars,
            known: vec![false; ni],
            terrain: vec![3; ni],
            owner: vec![0; ni],
            structure: vec![0; ni],
            resource: vec![0; ni],
            road: vec![false; ni],
            capital: vec![0; ni],
            units: Vec::new(),
            own: Vec::new(),
            enemy_cities: Vec::new(),
            neutral_villages: Vec::new(),
            tech: [false; 128],
            peace: false,
            met: false,
            actions: Vec::new(),
            enemy_near_city: 99,
            unit_targets: Vec::new(),
            city_threat: Vec::new(),
            my_units: 0,
            enemy_units: 0,
            unknown: 0,
            forest_own: 0,
            crop_own: 0,
            metal_own: 0,
            fruit_own: 0,
        };
        if let Some(ts) = vs["tiles"].as_array() {
            for t in ts {
                let i = t["tile"].as_i64().unwrap_or(0) as usize;
                if i >= ni {
                    continue;
                }
                let k = t["known"].as_bool().unwrap_or(false);
                w.known[i] = k;
                if k {
                    w.terrain[i] = t["terrain"].as_i64().unwrap_or(3) as i32;
                    w.owner[i] = t["owner"].as_i64().unwrap_or(0) as i32;
                    w.structure[i] = t["structure"].as_i64().unwrap_or(0) as i32;
                    w.resource[i] = t["resource"].as_i64().unwrap_or(0) as i32;
                    w.road[i] = t["road"].as_i64().unwrap_or(0) != 0;
                    w.capital[i] = t["capital"].as_i64().unwrap_or(0) as i32;
                }
            }
        }
        if let Some(us) = vs["units"].as_array() {
            for u in us {
                let kind = u["kind"].as_i64().unwrap_or(0) as i32;
                let veteran = u["veteran"].as_bool().unwrap_or(false);
                let st = stat(kind);
                let health = u["health"].as_f64().unwrap_or(10.0) as f32;
                w.units.push(Unit {
                    owner: u["owner"].as_i64().unwrap_or(0) as i32,
                    kind,
                    health,
                    maxhp: st.0 + if veteran { 5.0 } else { 0.0 },
                    tile: u["tile"].as_i64().unwrap_or(0) as i32,
                    attacked: u["attacked"].as_bool().unwrap_or(false),
                    moved: u["moved"].as_bool().unwrap_or(false),
                    kills: u["kills"].as_i64().unwrap_or(0) as i32,
                    veteran,
                });
            }
        }
        if let Some(cs) = vs["cities"].as_array() {
            for c in cs {
                let city = City {
                    tile: c["tile"].as_i64().unwrap_or(0) as i32,
                    owner: c["owner"].as_i64().unwrap_or(0) as i32,
                    level: c["level"].as_i64().unwrap_or(1) as i32,
                    production: c["production"].as_i64().unwrap_or(1) as i32,
                    progress: c["progress"].as_i64().unwrap_or(0) as i32,
                };
                if city.owner == player {
                    w.own.push(city);
                } else {
                    w.enemy_cities.push(city);
                }
            }
        }
        for i in 0..ni {
            if w.known[i] && w.structure[i] == 1 && w.owner[i] == 0 {
                w.neutral_villages.push(i as i32);
            }
        }
        if let Some(ts) = vs["technology"].as_array() {
            for t in ts {
                if let Some(k) = t.as_i64() {
                    if (0..128).contains(&k) {
                        w.tech[k as usize] = true;
                    }
                }
            }
        }
        let pi = (player - 1).max(0) as usize;
        let other = 1 - pi;
        if let Some(rel) = vs["relations"].as_object() {
            if let Some(p) = rel.get("peace").and_then(|v| v.as_array()) {
                if let Some(row) = p.get(pi).and_then(|v| v.as_array()) {
                    w.peace = row.get(other).and_then(|v| v.as_bool()).unwrap_or(false);
                }
            }
            if let Some(p) = rel.get("known_players").and_then(|v| v.as_array()) {
                if let Some(row) = p.get(pi).and_then(|v| v.as_array()) {
                    w.met = row.get(other).and_then(|v| v.as_bool()).unwrap_or(false);
                }
            }
        }
        if let Some(as_) = o["legal_actions"].as_array() {
            for a in as_ {
                w.actions.push(Act {
                    id: a["action_id"].as_str().unwrap_or("").to_string(),
                    kind: a["command"]["kind"].as_i64().unwrap_or(0) as i32,
                    source: a["command"]["source"].as_i64().unwrap_or(-1) as i32,
                    target: a["command"]["target"].as_i64().unwrap_or(-1) as i32,
                    param: a["command"]["parameter"].as_i64().unwrap_or(0) as i32,
                });
            }
        }
        // derived
        w.unknown = w.known.iter().filter(|&&k| !k).count() as i32;
        for i in 0..ni {
            if w.owned(i as i32) {
                if w.structure[i] == 0 && w.terrain[i] == 5 {
                    w.forest_own += 1;
                }
                if w.resource[i] == 2 {
                    w.crop_own += 1;
                }
                if w.resource[i] == 5 {
                    w.metal_own += 1;
                }
                if w.resource[i] == 6 {
                    w.fruit_own += 1;
                }
            }
        }
        let mut my = 0;
        let mut en = 0;
        let mut near = 99;
        for u in &w.units {
            if u.owner == w.player {
                my += 1;
            } else {
                en += 1;
                for c in &w.own {
                    near = near.min(w.d(u.tile, c.tile));
                }
            }
        }
        w.my_units = my;
        w.enemy_units = en;
        w.enemy_near_city = near;
        for c in &w.own {
            let mut t = 99;
            for u in &w.units {
                if u.owner != w.player {
                    t = t.min(w.d(c.tile, u.tile));
                }
            }
            w.city_threat.push((c.tile, t));
        }
        w.assign_targets();
        w
    }

    fn assign_targets(&mut self) {
        let own: Vec<usize> = (0..self.units.len()).filter(|&i| self.units[i].owner == self.player).collect();
        let mut used = vec![false; self.units.len()];
        let mut out: Vec<(i32, i32)> = Vec::new();
        // 1. defend against enemy units close to our cities
        for &i in &own {
            let u = &self.units[i];
            let mut best: Option<(i32, i32)> = None; // (dist, tile)
            for e in &self.units {
                if e.owner == self.player {
                    continue;
                }
                let mut dc = 99;
                for c in &self.own {
                    dc = dc.min(self.d(e.tile, c.tile));
                }
                if dc <= 3 {
                    let dd = self.d(u.tile, e.tile);
                    if best.map(|b| dd < b.0).unwrap_or(true) {
                        best = Some((dd, e.tile));
                    }
                }
            }
            if let Some((_, t)) = best {
                out.push((u.tile, t));
                used[i] = true;
            }
        }
        // 1b. keep one defender on each own city once the enemy is known
        if self.met {
            for c in &self.own {
                let threat = self.city_threat.iter().find(|&&(t, _)| t == c.tile).map(|&(_, d)| d).unwrap_or(99);
                if threat > 10 {
                    continue;
                }
                // the healthiest unit standing on this city stays
                let mut best: Option<(f32, usize)> = None;
                for &i in &own {
                    if used[i] || self.units[i].tile != c.tile {
                        continue;
                    }
                    let h = self.units[i].health;
                    if best.map(|b| h > b.0).unwrap_or(true) {
                        best = Some((h, i));
                    }
                }
                if let Some((_, i)) = best {
                    used[i] = true;
                }
            }
        }
        // 2. spread the rest over neutral villages
        let mut villages = self.neutral_villages.clone();
        loop {
            let mut best: Option<(i32, usize, usize)> = None; // dist, unit idx, village idx
            for &i in &own {
                if used[i] {
                    continue;
                }
                for (k, &v) in villages.iter().enumerate() {
                    let dd = self.d(self.units[i].tile, v);
                    if best.map(|b| dd < b.0).unwrap_or(true) {
                        best = Some((dd, i, k));
                    }
                }
            }
            match best {
                Some((_, i, k)) => {
                    out.push((self.units[i].tile, villages[k]));
                    used[i] = true;
                    villages.remove(k);
                }
                None => break,
            }
        }
        // 3. attack enemy cities once we have an army
        if !self.peace && self.my_units >= 6 {
            let mut cities: Vec<i32> = self.enemy_cities.iter().map(|c| c.tile).collect();
            loop {
                let mut best: Option<(i32, usize, usize)> = None;
                for &i in &own {
                    if used[i] {
                        continue;
                    }
                    for (k, &v) in cities.iter().enumerate() {
                        let dd = self.d(self.units[i].tile, v);
                        if best.map(|b| dd < b.0).unwrap_or(true) {
                            best = Some((dd, i, k));
                        }
                    }
                }
                match best {
                    Some((_, i, k)) => {
                        out.push((self.units[i].tile, cities[k]));
                        used[i] = true;
                        cities.remove(k);
                    }
                    None => break,
                }
            }
        }
        self.unit_targets = out;
    }

    #[inline]
    fn d(&self, a: i32, b: i32) -> i32 {
        let (ax, ay) = (a % self.size, a / self.size);
        let (bx, by) = (b % self.size, b / self.size);
        (ax - bx).abs().max((ay - by).abs())
    }

    #[inline]
    fn owned(&self, i: i32) -> bool {
        i >= 0 && (i as usize) < self.known.len() && self.owner[i as usize] == self.player
    }

    fn at(&self, tile: i32) -> Option<&Unit> {
        self.units.iter().find(|u| u.tile == tile && u.owner == self.player)
    }

    fn enemy_at(&self, tile: i32) -> Option<&Unit> {
        self.units.iter().find(|u| u.tile == tile && u.owner != self.player)
    }

    fn def_bonus(&self, tile: i32) -> f32 {
        let i = tile as usize;
        if i >= self.known.len() {
            return 1.0;
        }
        if self.structure[i] == 1 {
            1.5
        } else if self.terrain[i] == 5 {
            1.2
        } else if self.terrain[i] == 4 {
            1.2
        } else {
            1.0
        }
    }

    // ---------- main selection ----------
    fn choose(&self) -> String {
        let mut best = f64::NEG_INFINITY;
        let mut bid = String::new();
        for a in &self.actions {
            let s = self.score(a);
            if s > best {
                best = s;
                bid = a.id.clone();
            }
        }
        if bid.is_empty() {
            // fallback: end turn if present, else first
            for a in &self.actions {
                if a.kind == 10 {
                    return a.id.clone();
                }
            }
            if let Some(a) = self.actions.first() {
                return a.id.clone();
            }
        }
        bid
    }

    fn score(&self, a: &Act) -> f64 {
        match a.kind {
            10 => 0.0,
            11 => -1.0e9,
            9 => self.reward_score(a.param),
            8 => 1000000.0,
            2 => self.attack_score(a),
            4 => self.summon_score(a),
            7 => self.research_score(a.param),
            5 => 1380.0,
            6 => self.build_score(a.param, a.target),
            3 => self.ability_score(a),
            1 => self.move_score(a),
            _ => -1.0e6,
        }
    }

    fn reward_score(&self, p: i32) -> f64 {
        let threatened = self.enemy_near_city <= 5;
        match p {
            3 => 8300.0,      // workshop: +1 production
            4 => {
                if self.unknown * 3 > self.n {
                    9000.0
                } else {
                    700.0
                }
            }
            1 => {
                if threatened {
                    8900.0
                } else {
                    6000.0
                }
            }
            7 => 7600.0,      // +5 stars
            5 => 7000.0,      // border growth
            8 => 8200.0,      // population
            2 => 7900.0,      // park
            6 => 8600.0,      // giant
            _ => 5000.0,
        }
    }

    fn attack_score(&self, a: &Act) -> f64 {
        let src = a.source;
        let dst = a.target;
        let u = match self.at(src) {
            Some(u) => u,
            None => return -1.0e6,
        };
        let e = match self.enemy_at(dst) {
            Some(e) => e,
            None => return -1.0e6,
        };
        let (ahp, aatk, _, _, _, _) = stat(u.kind);
        let (dhp, _, ddef, _, _, _) = stat(e.kind);
        let _ = ahp;
        let _ = dhp;
        let am = u.maxhp.max(1.0);
        let dm = e.maxhp.max(1.0);
        let (dmg, mut ret) = combat(aatk, u.health, am, ddef, e.health, dm, self.def_bonus(dst));
        let dist = self.d(src, dst);
        if dist > stat(e.kind).3 {
            ret = 0.0;
        }
        let kill = dmg >= e.health;
        let d = dmg as f64;
        let r = ret as f64;
        let mut s: f64 = if r <= 0.0 {
            420.0 + d * 22.0
        } else if kill {
            5000.0
        } else if d > r {
            380.0 + d * 18.0 - r * 12.0
        } else {
            180.0 + d * 10.0 - r * 16.0
        };
        if kill {
            let value = match e.kind {
                12 | 45 => 900.0,
                11 | 4 => 250.0,
                5 => 220.0,
                9 => 220.0,
                _ => 150.0,
            };
            s += 3200.0 + value as f64;
            // killing a defender on an enemy city sets up a capture
            if self.structure[dst as usize] == 1 {
                s += 1500.0;
            }
        }
        if ret >= u.health {
            s -= 2200.0;
        }
        if u.health <= 4.0 && !kill {
            s -= 2600.0;
        }
        // enemy unit threatening our cities: worth more
        for c in &self.own {
            if self.d(dst, c.tile) <= 3 {
                s += 350.0;
                break;
            }
        }
        if dmg <= 1.0 && !kill {
            s -= 400.0;
        }
        s
    }

    fn summon_score(&self, a: &Act) -> f64 {
        let kind = a.param;
        let slots: i32 = self.own.iter().map(|c| c.level.max(1)).sum();
        let need = if self.enemy_near_city <= 6 {
            2700.0
        } else if self.my_units < slots + 1 {
            1650.0
        } else if self.my_units < self.enemy_units + 2 {
            1550.0
        } else if self.stars >= 8 {
            1150.0
        } else {
            return -80.0;
        };
        let pref = match kind {
            5 => 260.0,   // defender
            9 => 190.0,   // archer
            2 => 110.0,   // warrior
            3 => 100.0,   // rider
            11 => 240.0,  // swordsman
            4 => 220.0,   // knight
            12 => 260.0,  // giant
            8 => 150.0,   // catapult
            43 => -300.0, // raft
            _ => 0.0,
        };
        need + pref
    }

    fn research_score(&self, t: i32) -> f64 {
        let base = match t {
            18 => 1600.0, // archery
            7 => 1500.0,  // strategy (defender)
            16 => 1150.0 + (self.forest_own.min(6) as f64) * 80.0,
            8 => 1100.0 + (self.crop_own.min(6) as f64) * 100.0,
            6 => 1000.0 + (self.fruit_own.min(4) as f64) * 100.0,
            17 => 1150.0, // mathematics (sawmill, catapult)
            1 => 1000.0,  // riding
            20 => 950.0,  // climbing
            23 => 950.0 + (self.metal_own.min(4) as f64) * 100.0,
            24 => 1050.0, // smithery (swordsman)
            9 => 850.0,   // construction (windmill)
            3 => 850.0,   // chivalry (knight)
            10 => 850.0,  // fishing
            13 => 650.0,  // sailing
            44 => 300.0,
            12 => 300.0,
            2 => 200.0,
            4 => 200.0,
            5 => 200.0,
            19 => 200.0,
            21 => 200.0,
            22 => 200.0,
            38 => 200.0,
            121 => 200.0,
            _ => 150.0,
        };
        if self.turn < 40 && self.stars < 10 {
            -20.0
        } else {
            base
        }
    }

    fn build_score(&self, kind: i32, target: i32) -> f64 {
        let mut s = match kind {
            12 => 1360.0,                                  // lumber hut
            5 => 1400.0,                                   // farm
            21 => 1400.0,                                  // mine
            13 => 1050.0,                                  // sawmill
            22 => 1050.0,                                  // forge
            6 => 1000.0,                                   // windmill
            17 | 18 | 19 | 20 => 900.0,                    // temple
            3 => 480.0,                                    // road
            48 => 420.0,                                   // bridge
            8 => {
                if self.waterish() {
                    950.0
                } else {
                    200.0
                }
            }
            50 => 720.0,                                   // market
            23..=29 | 123 => 2300.0,                       // monuments (free, +3 pop)
            _ => 320.0,
        };
        if self.turn < 40 && self.stars < 7 && !(23..=29).contains(&kind) && kind != 123 {
            return -20.0;
        }
        // a population building belongs to a city that is close to levelling
        if matches!(kind, 5 | 12 | 21 | 13 | 22 | 6) {
            for c in &self.own {
                if self.d(target, c.tile) <= 2 && c.progress + 2 >= c.level + 1 {
                    s += 120.0;
                    break;
                }
            }
        }
        if kind == 8 && self.enemy_units > 0 {
            s += 150.0;
        }
        s
    }

    fn waterish(&self) -> bool {
        // true when a decent share of known tiles is water
        let mut water = 0;
        let mut tot = 0;
        for i in 0..self.n as usize {
            if self.known[i] {
                tot += 1;
                if self.terrain[i] <= 2 {
                    water += 1;
                }
            }
        }
        tot > 0 && water * 100 / tot >= 18
    }

    fn ability_score(&self, a: &Act) -> f64 {
        match a.param {
            16 => 6000.0, // promote
            7 => {
                let u = match self.at(a.source) {
                    Some(u) => u,
                    None => return -1.0e6,
                };
                if u.health + 3.0 < u.maxhp {
                    750.0
                } else {
                    -10.0
                }
            }
            9 => 300.0,
            1 => 260.0, // burn forest -> field with crop
            2 => 60.0,  // clear forest
            3 => 420.0, // grow forest
            8 => -5000.0,
            4 => -3000.0,
            18 => -9000.0, // break peace
            19 => {
                if a.target == 1 {
                    if self.turn >= 90 && !self.near_win() {
                        5000.0
                    } else {
                        -200.0
                    }
                } else {
                    -400.0
                }
            }
            20 => 220.0,
            21 => -5000.0,
            22 => -200.0,
            _ => 0.0,
        }
    }

    fn near_win(&self) -> bool {
        if self.enemy_cities.len() != 1 {
            return false;
        }
        let c = self.enemy_cities[0].tile;
        self.units
            .iter()
            .any(|u| u.owner == self.player && self.d(u.tile, c) <= 2)
    }

    fn move_score(&self, a: &Act) -> f64 {
        let src = a.source;
        let dst = a.target;
        let u = match self.at(src) {
            Some(u) => u,
            None => return -1.0e6,
        };
        let mut s = 0.0;
        // reveal new tiles around the destination
        let r = if self.terrain.get(dst as usize).copied().unwrap_or(3) == 4 {
            2
        } else {
            1
        };
        let mut newt = 0;
        let dx = dst % self.size;
        let dy = dst / self.size;
        for y in (dy - r).max(0)..=(dy + r).min(self.size - 1) {
            for x in (dx - r).max(0)..=(dx + r).min(self.size - 1) {
                let i = (y * self.size + x) as usize;
                if !self.known[i] {
                    newt += 1;
                }
            }
        }
        if newt > 0 {
            s += 65.0 + 22.0 * newt as f64;
        }
        if let Some(t) = self.unit_target(u) {
            let db = self.d(src, t);
            let da = self.d(dst, t);
            if da < db {
                s += 75.0 + 28.0 * (db - da) as f64;
            } else if da > db {
                s -= 35.0;
            }
        }
        // do not abandon a threatened city
        if self.structure.get(src as usize).copied().unwrap_or(0) == 1 && self.owned(src) {
            let mut threat = 99;
            for &(t, d) in &self.city_threat {
                if t == src {
                    threat = d;
                }
            }
            if threat <= 8 {
                s -= 1200.0;
            }
        }
        // step onto an undefended enemy city
        if self.structure.get(dst as usize).copied().unwrap_or(0) == 1
            && !self.owned(dst)
            && self.enemy_at(dst).is_none()
        {
            s += 220.0;
        }
        // moving into one of our cities to garrison it
        if self.structure.get(dst as usize).copied().unwrap_or(0) == 1
            && self.owned(dst)
            && self.enemy_near_city <= 7
        {
            if self.units.iter().any(|x| x.owner == self.player && x.tile == dst) {
                s -= 60.0;
            } else {
                s += 380.0;
            }
        }
        // prefer defensive terrain slightly
        let ter = self.terrain.get(dst as usize).copied().unwrap_or(3);
        if ter == 5 || ter == 4 {
            s += 12.0;
        }
        s
    }

    fn unit_target(&self, u: &Unit) -> Option<i32> {
        for &(tile, t) in &self.unit_targets {
            if tile == u.tile {
                return Some(t);
            }
        }
        None
    }
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
            Err(_) => break,
        };
        let w = W::parse(&obs);
        let id = w.choose();
        writeln!(stdout, "{}", json!({ "action_id": id })).unwrap();
        stdout.flush().unwrap();
    }
}
