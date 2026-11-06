// Bardur Polytopia 1v1 heuristic bot.
// One JSON observation per line; reply {"action_id":"..."} and flush.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone, Copy, Default)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    res: i32,
    st: i32,
    cap: i32,
}

#[derive(Clone, Copy, Default)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    hp: f32,
    moved: bool,
    attacked: bool,
    home: i32,
    vet: bool,
}

#[derive(Clone, Copy, Default)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    prod: i32,
    progress: i32,
    pop: i32,
}

#[derive(Clone)]
struct Act {
    id: String,
    kind: i32,
    src: i32,
    tgt: i32,
    par: i32,
}

struct Obs {
    size: usize,
    n: usize,
    turn: i32,
    stars: i32,
    me: i32,
    tech: [bool; 128],
    tiles: Vec<Tile>,
    units: Vec<Unit>,
    cities: Vec<City>,
    legal: Vec<Act>,
    board_units: Vec<i32>, // kind of unit on tile (0 = none)
    board_owner: Vec<i32>, // owner of unit on tile
}

// ---------- unit tables ----------
// (attack, defense, health, range, cost)
fn stat(kind: i32) -> (f32, f32, f32, i32, i32) {
    match kind {
        2 => (2.0, 2.0, 10.0, 1, 2),
        3 => (2.0, 1.0, 10.0, 1, 3),
        4 => (3.5, 1.0, 10.0, 1, 8),
        5 => (1.0, 3.0, 15.0, 1, 3),
        8 => (4.0, 0.0, 10.0, 3, 8),
        9 => (2.0, 1.0, 10.0, 2, 3),
        10 => (0.0, 1.0, 10.0, 1, 5),
        11 => (3.0, 3.0, 15.0, 1, 5),
        12 => (5.0, 4.0, 40.0, 1, 10),
        15 => (3.0, 1.0, 15.0, 1, 2),
        17 => (-1.0, 2.0, 10.0, 1, 10),
        18 => (3.0, 3.0, 15.0, 1, 10),
        19 => (4.0, 3.0, 20.0, 2, 10),
        20 => (2.0, 1.0, 10.0, 1, 3),
        21 => (2.5, 1.0, 10.0, 2, 8),
        22 => (0.0, 1.0, 10.0, 1, 5),
        23 => (3.0, 2.0, 15.0, 1, 5),
        24 => (4.0, 3.0, 20.0, 2, 15),
        25 => (1.0, 1.0, 10.0, 2, 3),
        26 => (4.0, 5.0, 40.0, 1, 10),
        27 => (4.0, 3.0, 30.0, 1, 10),
        28 => (3.0, 1.0, 5.0, 1, 3),
        29 => (3.5, 2.0, 20.0, 1, 10),
        30 => (0.7, 1.0, 5.0, 2, 3),
        31 => (1.0, 3.0, 15.0, 1, 3),
        32 => (3.0, 1.0, 10.0, 3, 8),
        33 => (4.0, 3.0, 20.0, 1, 10),
        34 => (2.0, 2.0, 10.0, 1, 10),
        35 => (3.0, 2.0, 10.0, 1, 5),
        36 => (1.0, 1.0, 10.0, 1, 10),
        37 => (2.0, 2.0, 10.0, 1, 2),
        38 => (0.0, 0.5, 5.0, 1, 8),
        39 => (0.0, 0.5, 5.0, 1, 2),
        40 => (2.0, 2.0, 10.0, 1, 2),
        41 => (3.0, 2.0, 10.0, 3, 15),
        42 => (2.0, 1.0, 10.0, 2, 5),
        43 => (0.0, 1.0, 10.0, 2, 0),
        44 => (3.0, 3.0, 10.0, 1, 5),
        45 => (4.0, 4.0, 40.0, 1, 10),
        56 => (3.0, 3.0, 10.0, 1, 5),
        57 => (4.0, 4.0, 20.0, 1, 20),
        59 => (3.0, 3.0, 15.0, 1, 5),
        60 => (2.0, 3.0, 10.0, 1, 0),
        61 => (2.0, 0.1, 10.0, 1, 5),
        62 => (2.0, 2.0, 10.0, 1, 0),
        _ => (1.0, 1.0, 10.0, 1, 2),
    }
}

fn is_stiff(kind: i32) -> bool {
    matches!(kind, 8 | 10 | 17 | 22 | 25 | 34 | 38 | 39 | 41 | 43 | 45 | 56 | 60 | 61)
}

fn is_naval(kind: i32) -> bool {
    matches!(kind, 39 | 40 | 41 | 42 | 43 | 44 | 45)
}

fn has_dash(kind: i32) -> bool {
    matches!(kind, 2 | 3 | 4 | 9 | 11 | 20 | 21 | 23 | 28 | 29 | 30 | 35 | 37 | 40 | 42 | 44 | 56 | 59 | 61 | 62)
}

fn combat_dmg(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32) {
    let af = aa * (ah / if am > 1.0 { am } else { 1.0 });
    let df = dd * (dh / if dm > 1.0 { dm } else { 1.0 }) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001 + 0.5)
        .floor();
    (attack, defense)
}

fn num(v: &Value, k: &str) -> i32 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32
}

fn parse(v: &Value) -> Option<Obs> {
    let vs = v.get("visible_state")?;
    let size = num(vs, "board_size") as usize;
    if size == 0 || size > 40 {
        return None;
    }
    let n = size * size;
    let mut tiles = vec![Tile::default(); n];
    if let Some(arr) = vs.get("tiles").and_then(|x| x.as_array()) {
        for t in arr {
            let i = num(t, "tile") as usize;
            if i >= n {
                continue;
            }
            let known = t.get("known").and_then(|x| x.as_bool()).unwrap_or(false);
            tiles[i] = Tile {
                known,
                terrain: num(t, "terrain"),
                owner: num(t, "owner"),
                res: num(t, "resource"),
                st: num(t, "structure"),
                cap: num(t, "capital"),
            };
        }
    }
    let mut units = Vec::new();
    let mut board_units = vec![0i32; n];
    let mut board_owner = vec![0i32; n];
    if let Some(arr) = vs.get("units").and_then(|x| x.as_array()) {
        for u in arr {
            let tile = num(u, "tile") as usize;
            if tile >= n {
                continue;
            }
            let kind = num(u, "kind");
            let owner = num(u, "owner");
            units.push(Unit {
                tile,
                owner,
                kind,
                hp: u.get("health").and_then(|x| x.as_f64()).unwrap_or(10.0) as f32,
                moved: u.get("moved").and_then(|x| x.as_bool()).unwrap_or(false),
                attacked: u.get("attacked").and_then(|x| x.as_bool()).unwrap_or(false),
                home: num(u, "home"),
                vet: u.get("veteran").and_then(|x| x.as_bool()).unwrap_or(false),
            });
            board_units[tile] = kind;
            board_owner[tile] = owner;
        }
    }
    let mut cities = Vec::new();
    if let Some(arr) = vs.get("cities").and_then(|x| x.as_array()) {
        for c in arr {
            let tile = num(c, "tile") as usize;
            if tile >= n {
                continue;
            }
            cities.push(City {
                tile,
                owner: num(c, "owner"),
                level: num(c, "level"),
                prod: num(c, "production"),
                progress: num(c, "progress"),
                pop: num(c, "population"),
            });
        }
    }
    let mut tech = [false; 128];
    if let Some(arr) = vs.get("technology").and_then(|x| x.as_array()) {
        for t in arr {
            if let Some(k) = t.as_i64() {
                if k >= 0 && k < 128 {
                    tech[k as usize] = true;
                }
            }
        }
    }
    let mut legal = Vec::new();
    if let Some(arr) = v.get("legal_actions").and_then(|x| x.as_array()) {
        for a in arr {
            let id = a.get("action_id").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let c = a.get("command");
            let (kind, src, tgt, par) = match c {
                Some(c) => (num(c, "kind"), num(c, "source"), num(c, "target"), num(c, "parameter")),
                None => (0, -1, -1, 0),
            };
            legal.push(Act { id, kind, src, tgt, par });
        }
    }
    Some(Obs {
        size,
        n,
        turn: num(vs, "turn"),
        stars: num(vs, "stars"),
        me: num(v, "player"),
        tech,
        tiles,
        units,
        cities,
        legal,
        board_units,
        board_owner,
    })
}

// ---------- bot state ----------
struct Bot {
    n: usize,
    terrain: Vec<i32>,
    st: Vec<i32>,
    res: Vec<i32>,
    last_turn: i32,
    tech_seq: Vec<i32>,
}

fn default_tech_seq() -> Vec<i32> {
    vec![1, 6, 18, 16, 8, 9, 17, 20, 23, 24, 7, 4, 10, 13, 5]
}

impl Bot {
    fn new() -> Bot {
        Bot { n: 0, terrain: Vec::new(), st: Vec::new(), res: Vec::new(), last_turn: -1, tech_seq: default_tech_seq() }
    }

    fn refresh(&mut self, o: &Obs) {
        if self.n != o.n || o.turn < self.last_turn || (o.turn == 0 && self.last_turn >= 0) {
            self.n = o.n;
            self.terrain = vec![0; o.n];
            self.st = vec![0; o.n];
            self.res = vec![0; o.n];
        }
        for (i, t) in o.tiles.iter().enumerate() {
            if t.known {
                self.terrain[i] = t.terrain;
                self.st[i] = t.st;
                self.res[i] = t.res;
            }
        }
        self.last_turn = o.turn;
    }

    fn choose(&mut self, o: &Obs) -> String {
        self.refresh(o);
        if o.legal.is_empty() {
            return String::new();
        }
        if let Some(id) = self.pick_reward(o) {
            return id;
        }
        if let Some(a) = o.legal.iter().find(|a| a.kind == 3 && a.par == 16) {
            return a.id.clone();
        }
        if let Some(id) = self.pick_capture(o) {
            return id;
        }
        if let Some(id) = self.pick_attack(o) {
            return id;
        }
        let my_cities = o.cities.iter().filter(|c| c.owner == o.me).count() as i32;
        let my_units = o.units.iter().filter(|u| u.owner == o.me).count() as i32;
        let enemy_near = self.enemy_pressure(o) > 0;
        let desired = self.desired_units(o, my_cities, enemy_near);
        let margin = if my_units < desired { 3 } else { 0 };

        if let Some(id) = self.pick_defense_summon(o) {
            return id;
        }
        if let Some(id) = self.pick_harvest(o, my_units, desired) {
            return id;
        }
        if let Some(id) = self.pick_move(o) {
            return id;
        }
        if my_units < desired {
            if let Some(id) = self.pick_summon(o, my_cities) {
                return id;
            }
        }
        if let Some(id) = self.pick_research(o, my_units, desired, true) {
            return id;
        }
        if let Some(id) = self.pick_research(o, my_units, desired, false) {
            return id;
        }
        if let Some(id) = self.pick_build(o, margin) {
            return id;
        }
        if let Some(a) = o.legal.iter().find(|a| a.kind == 10) {
            return a.id.clone();
        }
        o.legal[0].id.clone()
    }

    fn pick_reward(&self, o: &Obs) -> Option<String> {
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 9 {
                continue;
            }
            let v = match a.par {
                3 => 88.0,
                4 => 56.0,
                1 => {
                    if self.enemy_pressure(o) > 0 {
                        97.0
                    } else {
                        20.0
                    }
                }
                7 => 95.0,
                5 => 52.0,
                8 => 90.0,
                2 => 70.0,
                6 => {
                    if o.turn < 15 {
                        12.0
                    } else {
                        82.0
                    }
                }
                _ => 10.0,
            };
            if best.as_ref().map_or(true, |(bv, _)| v > *bv) {
                best = Some((v, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn pick_capture(&self, o: &Obs) -> Option<String> {
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 8 {
                continue;
            }
            let src = a.src;
            if src < 0 || src as usize >= o.n {
                continue;
            }
            let ti = o.tiles[src as usize];
            let v = if ti.st == 1 {
                if ti.owner > 0 && ti.owner != o.me {
                    if ti.cap == ti.owner {
                        10000.0
                    } else {
                        9000.0
                    }
                } else {
                    5000.0
                }
            } else if ti.st == 2 {
                3000.0
            } else {
                1500.0
            };
            if best.as_ref().map_or(true, |(bv, _)| v > *bv) {
                best = Some((v, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn def_bonus(&self, o: &Obs, t: usize) -> f32 {
        if t >= o.n {
            return 1.0;
        }
        let ti = o.tiles[t];
        if ti.st == 1 {
            1.5
        } else {
            match ti.terrain {
                4 | 5 => 1.5,
                _ => 1.0,
            }
        }
    }

    fn pick_attack(&self, o: &Obs) -> Option<String> {
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 2 {
                continue;
            }
            let (src, tgt) = (a.src, a.tgt);
            if src < 0 || tgt < 0 || src as usize >= o.n || tgt as usize >= o.n {
                continue;
            }
            let (su, tu) = (src as usize, tgt as usize);
            let ak = o.board_units[su];
            let dk = o.board_units[tu];
            if ak == 0 || dk == 0 {
                continue;
            }
            let aown = o.board_owner[su];
            // find our unit hp
            let au = o.units.iter().find(|u| u.tile == su && u.owner == aown);
            let du = o.units.iter().find(|u| u.tile == tu && u.owner == o.board_owner[tu]);
            let (ah, _av) = match au {
                Some(u) => (u.hp, u.vet),
                None => (stat(ak).2, false),
            };
            let dh = match du {
                Some(u) => u.hp,
                None => stat(dk).2,
            };
            let (aa, _ad, amax, _ar, acost) = stat(ak);
            let (da, dd, dmax, drange, dcost) = stat(dk);
            let amax = amax + if au.map(|u| u.vet).unwrap_or(false) { 5.0 } else { 0.0 };
            let bonus = self.def_bonus(o, tu);
            let (dmg, ret) = combat_dmg(aa, ah, amax, dd, dh, dmax, bonus);
            let kill = dmg >= dh;
            let ret_ok = !kill && !is_stiff(dk) && dist(o.size, su, tu) <= drange && da >= 0.0;
            let ret2 = if ret_ok { ret } else { 0.0 };
            let mut score = dmg * 2.0;
            if kill {
                score += 20.0 + 6.0 * dcost as f32;
            }
            score -= ret2 * 3.0 * (acost as f32 / dcost.max(1) as f32);
            if ret2 >= ah && !kill {
                score -= 60.0;
            }
            if o.tiles[tu].st == 1 {
                score += 2.0;
            }
            if best.as_ref().map_or(true, |(bv, _)| score > *bv) {
                best = Some((score, a.id.clone()));
            }
        }
        match best {
            Some((s, id)) if s > 3.0 => Some(id),
            _ => None,
        }
    }

    fn pick_research(&self, o: &Obs, my_units: i32, desired: i32, top_only: bool) -> Option<String> {
        let mut best: Option<(usize, String)> = None;
        for a in &o.legal {
            if a.kind != 7 {
                continue;
            }
            let tech = a.par;
            let rank = match self.tech_seq.iter().position(|&t| t == tech) {
                Some(r) => r,
                None => continue,
            };
            if top_only && rank > 0 {
                continue;
            }
            if best.as_ref().map_or(true, |(br, _)| rank < *br) {
                best = Some((rank, a.id.clone()));
            }
        }
        match best {
            Some((rank, id)) => {
                // keep some money for units when the army is short
                if top_only {
                    return Some(id);
                }
                if o.stars < 12 {
                    return None;
                }
                if my_units < desired && rank >= 2 && o.stars < 16 {
                    return None;
                }
                Some(id)
            }
            None => None,
        }
    }

    fn pick_harvest(&self, o: &Obs, my_units: i32, desired: i32) -> Option<String> {
        if o.stars < 2 {
            return None;
        }
        let army_short = my_units < desired;
        if army_short && o.stars < 5 {
            return None;
        }
        // prefer crops we will not farm, then animals/fish anywhere
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 5 {
                continue;
            }
            let t = a.tgt;
            if t < 0 || t as usize >= o.n {
                continue;
            }
            let ti = o.tiles[t as usize];
            let v = match ti.res {
                1 => 3.0,
                3 => 3.0,
                6 => {
                    if o.tech[8] {
                        1.0
                    } else {
                        2.5
                    }
                }
                _ => 1.0,
            };
            if best.as_ref().map_or(true, |(bv, _)| v > *bv) {
                best = Some((v, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn pick_summon(&self, o: &Obs, my_cities: i32) -> Option<String> {
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 4 {
                continue;
            }
            let k = a.par;
            let base = match k {
                11 => 100.0,
                4 => 95.0,
                8 => 92.0,
                12 => 85.0,
                3 => 72.0,
                9 => 62.0,
                5 => 58.0,
                2 => 50.0,
                _ => 0.0,
            };
            if base <= 0.0 {
                continue;
            }
            let mut score = base;
            if k == 3 && o.turn < 45 {
                score += 18.0;
            }
            if k == 11 && o.turn < 25 {
                score -= 10.0;
            }
            // prefer cheap units when poor
            let cost = stat(k).4;
            if o.stars < cost + 6 {
                score -= cost as f32 * 1.5;
            }
            if my_cities <= 1 && k == 12 {
                score -= 20.0;
            }
            if best.as_ref().map_or(true, |(bv, _)| score > *bv) {
                best = Some((score, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn build_cost(kind: i32) -> i32 {
        match kind {
            3 => 3,
            5 => 5,
            6 => 5,
            8 => 7,
            12 => 3,
            13 => 5,
            21 => 5,
            22 => 5,
            50 => 5,
            17 => 20,
            18 => 15,
            19 => 20,
            20 => 20,
            _ => 99,
        }
    }

    fn pick_build(&self, o: &Obs, margin: i32) -> Option<String> {
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 6 {
                continue;
            }
            let k = a.par;
            let cost = Bot::build_cost(k);
            if o.stars < cost + margin.max(0) {
                continue;
            }
            let v = match k {
                13 | 6 | 22 => 30.0,
                50 => 26.0,
                5 | 21 => 24.0,
                12 => 18.0,
                8 => 8.0,
                3 => 2.0,
                _ => -1.0,
            };
            if v <= 0.0 {
                continue;
            }
            let mut score = v - cost as f32 * 0.4;
            if o.turn > 150 && k == 50 {
                score -= 10.0;
            }
            if best.as_ref().map_or(true, |(bv, _)| score > *bv) {
                best = Some((score, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn enemy_pressure(&self, o: &Obs) -> i32 {
        let mut count = 0;
        for u in &o.units {
            if u.owner == 0 || u.owner == o.me {
                continue;
            }
            for c in &o.cities {
                if c.owner != o.me {
                    continue;
                }
                if dist(o.size, u.tile, c.tile) <= 5 {
                    count += 1;
                    break;
                }
            }
        }
        count
    }

    fn desired_units(&self, o: &Obs, my_cities: i32, enemy_near: bool) -> i32 {
        let mut base = 3 * my_cities + 2 + o.turn / 8;
        if enemy_near {
            base += 2;
        }
        base.min(18).max(3)
    }

    fn dist_field(&self, o: &Obs, cost: &[i32], seeds: &[(usize, i32)]) -> Vec<i32> {
        let n = o.n;
        let mut d = vec![INF; n];
        for &(s, c) in seeds {
            if s < n && c < d[s] {
                d[s] = c;
            }
        }
        let size = o.size as i32;
        let mut changed = true;
        let mut guard = 0;
        while changed && guard < n + 4 {
            changed = false;
            guard += 1;
            for i in 0..n {
                let di = d[i];
                if di >= INF {
                    continue;
                }
                let (x, y) = ((i % o.size) as i32, (i / o.size) as i32);
                for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || ny < 0 || nx >= size || ny >= size {
                            continue;
                        }
                        let j = (ny as usize) * o.size + nx as usize;
                        let c = cost[j];
                        if c >= INF {
                            continue;
                        }
                        let nd = di.saturating_add(c);
                        if nd < d[j] {
                            d[j] = nd;
                            changed = true;
                        }
                    }
                }
            }
        }
        d
    }

    fn pick_defense_summon(&self, o: &Obs) -> Option<String> {
        // an own city with an enemy on or near it and no garrison: put a cheap body there
        let mut threatened: Vec<usize> = Vec::new();
        for c in &o.cities {
            if c.owner != o.me {
                continue;
            }
            let near = o.units.iter().any(|u| {
                u.owner != 0 && u.owner != o.me && dist(o.size, u.tile, c.tile) <= 4
            });
            if near && o.board_units[c.tile] == 0 {
                threatened.push(c.tile);
            }
        }
        if threatened.is_empty() {
            return None;
        }
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 4 {
                continue;
            }
            let t = a.src;
            if t < 0 || t as usize >= o.n {
                continue;
            }
            if !threatened.contains(&(t as usize)) {
                continue;
            }
            let k = a.par;
            // cheapest solid body
            let v = match k {
                5 => 90.0,
                2 => 80.0,
                11 => 85.0,
                9 => 70.0,
                3 => 60.0,
                _ => 0.0,
            };
            let v = v - stat(k).4 as f32 * 2.0;
            if best.as_ref().map_or(true, |(bv, _)| v > *bv) {
                best = Some((v, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn pick_move(&self, o: &Obs) -> Option<String> {
        let n = o.n;
        let pressure = self.enemy_pressure(o);
        let war = o.turn > 25 || pressure > 0;
        // ---- movement cost arrays
        let mut land_cost = vec![INF; n];
        let mut sea_cost = vec![INF; n];
        for t in 0..n {
            let tr = if o.tiles[t].known { o.tiles[t].terrain } else { self.terrain[t] };
            let water = tr == 1 || tr == 2;
            let mountain = tr == 4;
            sea_cost[t] = if water { 1 } else { INF };
            land_cost[t] = if water {
                INF
            } else if mountain {
                if o.tech[20] {
                    3
                } else {
                    INF
                }
            } else if tr == 5 {
                2
            } else {
                1
            };
        }
        // ---- goals: unowned villages, ruins, enemy cities
        let mut goals: Vec<(usize, i32)> = Vec::new();
        let mut is_goal = vec![false; n];
        for t in 0..n {
            let ti = o.tiles[t];
            if !ti.known {
                continue;
            }
            if ti.st == 1 && ti.owner == 0 {
                goals.push((t, 0));
                is_goal[t] = true;
            } else if ti.st == 1 && ti.owner > 0 && ti.owner != o.me {
                goals.push((t, if war { 0 } else { 8 }));
                is_goal[t] = true;
            } else if ti.st == 2 {
                goals.push((t, 2));
                is_goal[t] = true;
            }
        }
        let gf = self.dist_field(o, &land_cost, &goals);
        let unknown: Vec<usize> = (0..n).filter(|&t| !o.tiles[t].known).collect();
        // ---- threatened cities
        let mut threat = vec![false; n];
        for u in &o.units {
            if u.owner == 0 || u.owner == o.me {
                continue;
            }
            for c in &o.cities {
                if c.owner != o.me {
                    continue;
                }
                if dist(o.size, u.tile, c.tile) <= 4 {
                    threat[c.tile] = true;
                }
            }
        }
        // ---- danger: tiles next to a visible enemy unit
        let mut danger = vec![0i32; n];
        for u in &o.units {
            if u.owner == 0 || u.owner == o.me {
                continue;
            }
            let (x, y) = ((u.tile % o.size) as i32, (u.tile / o.size) as i32);
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= o.size as i32 || ny >= o.size as i32 {
                        continue;
                    }
                    danger[(ny as usize) * o.size + nx as usize] += 1;
                }
            }
        }
        // ---- tiles with an enemy within 3
        let mut near_enemy = vec![0i32; n];
        for u in &o.units {
            if u.owner == 0 || u.owner == o.me {
                continue;
            }
            let (x, y) = ((u.tile % o.size) as i32, (u.tile / o.size) as i32);
            for dy in -3..=3i32 {
                for dx in -3..=3i32 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= o.size as i32 || ny >= o.size as i32 {
                        continue;
                    }
                    near_enemy[(ny as usize) * o.size + nx as usize] += 1;
                }
            }
        }
        // ---- idle movable units of ours
        let mut movers: Vec<usize> = Vec::new();
        for a in &o.legal {
            if a.kind != 1 || a.src < 0 {
                continue;
            }
            let s = a.src as usize;
            if s < n && o.board_owner[s] == o.me && o.board_units[s] != 0 && !movers.contains(&s) {
                movers.push(s);
            }
        }
        if movers.is_empty() {
            return None;
        }
        // ---- assign a target tile to every mover
        let m = movers.len();
        let mut tgt_of: Vec<usize> = vec![usize::MAX; m];
        let mut goal_used = vec![false; n];
        // 1. defenders for threatened empty cities
        for c in 0..n {
            if !threat[c] || o.tiles[c].st != 1 || o.tiles[c].owner != o.me {
                continue;
            }
            if o.board_units[c] != 0 {
                continue;
            }
            let mut besti = usize::MAX;
            let mut bestd = 99;
            for (k, &s) in movers.iter().enumerate() {
                if tgt_of[k] != usize::MAX {
                    continue;
                }
                let d = dist(o.size, s, c);
                if d < bestd {
                    bestd = d;
                    besti = k;
                }
            }
            if besti != usize::MAX {
                tgt_of[besti] = c;
            }
        }
        // 2. goals, nearest-first per unit
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by_key(|&k| gf[movers[k]]);
        for &k in &order {
            if tgt_of[k] != usize::MAX || goals.is_empty() {
                continue;
            }
            let mut bg = usize::MAX;
            let mut bv = INF;
            for &(g, _) in &goals {
                if goal_used[g] {
                    continue;
                }
                let d = gf[movers[k]];
                let _ = d;
                let v = dist(o.size, movers[k], g);
                if v < bv {
                    bv = v;
                    bg = g;
                }
            }
            if bg != usize::MAX {
                goal_used[bg] = true;
                tgt_of[k] = bg;
            }
        }
        // 3. garrisons: own cities without a unit, capital and threatened first
        let mut threat_list: Vec<usize> = Vec::new();
        let mut own: Vec<usize> = o.cities.iter().filter(|c| c.owner == o.me).map(|c| c.tile).collect();
        own.sort_by_key(|&t| {
            let caprank = if o.tiles[t].cap == o.me { 0 } else { 1 };
            let threatrank = if threat[t] { 0 } else { 1 };
            (caprank, threatrank, t)
        });
        for &t in &own {
            if o.board_units[t] == 0 && (threat[t] || (o.tiles[t].cap == o.me && pressure > 0)) {
                threat_list.push(t);
            }
        }
        if !threat_list.is_empty() {
            for &k in &order {
                if tgt_of[k] != usize::MAX {
                    continue;
                }
                let s = movers[k];
                let mut bg = usize::MAX;
                let mut bv = 999;
                for &c in &threat_list {
                    let v = dist(o.size, s, c);
                    if v < bv {
                        bv = v;
                        bg = c;
                    }
                }
                tgt_of[k] = bg;
            }
        }
        // 4. scout: nearest unknown tile per unit, spread over directions
        let mut taken = vec![false; n];
        let mut used_sector = [false; 8];
        for &k in &order {
            if tgt_of[k] != usize::MAX {
                continue;
            }
            let s = movers[k];
            let (sx, sy) = ((s % o.size) as f32, (s / o.size) as f32);
            let mut bg = usize::MAX;
            let mut bv = 1e9f32;
            let mut bg_free = usize::MAX;
            let mut bv_free = 1e9f32;
            for &u in &unknown {
                if taken[u] {
                    continue;
                }
                let (ux0, uy0) = ((u % o.size) as f32, (u / o.size) as f32);
                let v = (ux0 - sx) * (ux0 - sx) + (uy0 - sy) * (uy0 - sy);
                if v < bv_free {
                    bv_free = v;
                    bg_free = u;
                }
                let (ux, uy) = ((u % o.size) as f32, (u / o.size) as f32);
                let ang = (uy - sy).atan2(ux - sx);
                let sec = (((ang + 3.14159265) / 0.78539816) as usize).min(7);
                if used_sector[sec] {
                    continue;
                }
                if v < bv {
                    bv = v;
                    bg = u;
                }
            }
            if bg != usize::MAX {
                let (ux, uy) = ((bg % o.size) as f32, (bg / o.size) as f32);
                let ang = (uy - sy).atan2(ux - sx);
                let sec = (((ang + 3.14159265) / 0.78539816) as usize).min(7);
                used_sector[sec] = true;
                taken[bg] = true;
                tgt_of[k] = bg;
            } else if bg_free != usize::MAX {
                taken[bg_free] = true;
                tgt_of[k] = bg_free;
            }
        }
        // ---- score moves
        let mut best: Option<(f32, String)> = None;
        for a in &o.legal {
            if a.kind != 1 || a.src < 0 || a.tgt < 0 {
                continue;
            }
            let (su, tu) = (a.src as usize, a.tgt as usize);
            if su >= n || tu >= n {
                continue;
            }
            let kind = o.board_units[su];
            if o.board_owner[su] != o.me || kind == 0 {
                continue;
            }
            let ki = match movers.iter().position(|&x| x == su) {
                Some(k) => k,
                None => continue,
            };
            let tgt = tgt_of[ki];
            if tgt == usize::MAX {
                continue;
            }
            // garrison: never leave a threatened city that we are holding
            if o.tiles[su].st == 1 && o.tiles[su].owner == o.me && (near_enemy[su] > 0 || (o.tiles[su].cap == o.me && pressure > 0)) {
                continue;
            }
            // non-dash units with a target in reach should stand and fight
            if !has_dash(kind) {
                let rng = stat(kind).3.max(1);
                let mut has_target = false;
                for u in &o.units {
                    if u.owner != 0 && u.owner != o.me && dist(o.size, su, u.tile) <= rng {
                        has_target = true;
                    }
                }
                if has_target {
                    continue;
                }
            }
            let metric = |t: usize| -> f32 {
                if is_goal[tgt] {
                    gf[t].min(INF) as f32
                } else {
                    dist(o.size, t, tgt) as f32
                }
            };
            let mut mv = -(metric(tu)) * 10.0;
            let stay = -(metric(su)) * 10.0;
            if !o.tiles[tu].known {
                mv += 1.5;
            }
            if threat[tu] {
                mv += 12.0;
            }
            let own_city = o.tiles[tu].st == 1 && o.tiles[tu].owner == o.me;
            if own_city && o.board_units[tu] == 0 {
                mv += 4.0;
            }
            // avoid walking into reach of the enemy
            if danger[tu] > danger[su] && !own_city {
                mv -= 24.0 * (danger[tu] - danger[su]) as f32;
            }
            if danger[tu] > 0 && danger[su] > 0 && danger[tu] < danger[su] {
                mv += 6.0;
            }
            // wounded units fall back to the nearest own city
            let hp = o.units.iter().find(|u| u.tile == su && u.owner == o.me).map(|u| u.hp).unwrap_or(10.0);
            if hp <= 6.0 && !own_city {
                let mut nd = 99;
                for &c in &own {
                    let d = dist(o.size, tu, c);
                    if d < nd {
                        nd = d;
                    }
                }
                let mut sd = 99;
                for &c in &own {
                    let d = dist(o.size, su, c);
                    if d < sd {
                        sd = d;
                    }
                }
                mv += (sd - nd) as f32 * 12.0;
            }
            if mv > stay + 0.01 {
                if best.as_ref().map_or(true, |(bv, _)| mv > *bv) {
                    best = Some((mv, a.id.clone()));
                }
            }
        }
        best.map(|(_, id)| id)
    }
}

fn dist(size: usize, a: usize, b: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

fn main() {
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    let mut bot = Bot::new();
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
        let id = match parse(&v) {
            Some(o) => bot.choose(&o),
            None => String::new(),
        };
        let id = if id.is_empty() {
            v.get("legal_actions")
                .and_then(|x| x.as_array())
                .and_then(|a| a.first())
                .and_then(|a| a.get("action_id"))
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string()
        } else {
            id
        };
        let _ = writeln!(out, "{}", json!({ "action_id": id }));
        let _ = out.flush();
    }
}
