// Polytopia Bardur 1v1 bot: heuristic priority policy over the legal action catalog.
use serde_json::Value;
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;
const F_DASH: u32 = 1;
const F_STIFF: u32 = 2;
const F_FLOAT: u32 = 4;
const F_CARRY: u32 = 8;
const F_SCOUT: u32 = 16;
const F_FORTIFY: u32 = 256;

fn ustat(k: i32) -> (i32, f32, f32, f32, i32, i32, u32) {
    // cost, hp, atk, def, movement, range, flags
    match k {
        2 => (2, 10.0, 2.0, 2.0, 1, 1, F_DASH | F_FORTIFY),
        3 => (3, 10.0, 2.0, 1.0, 2, 1, F_DASH | 32 | F_FORTIFY),
        4 => (8, 10.0, 3.5, 1.0, 3, 1, F_DASH | F_FORTIFY | 64),
        5 => (3, 15.0, 1.0, 3.0, 1, 1, F_FORTIFY),
        8 => (8, 10.0, 4.0, 0.0, 1, 3, F_STIFF),
        9 => (3, 10.0, 2.0, 1.0, 1, 2, F_DASH | F_FORTIFY),
        10 => (5, 10.0, 0.0, 1.0, 1, 1, F_STIFF),
        11 => (5, 15.0, 3.0, 3.0, 1, 1, F_DASH),
        12 => (10, 40.0, 5.0, 4.0, 1, 1, 512),
        15 => (2, 15.0, 3.0, 1.0, 1, 1, F_DASH | F_FORTIFY | 512),
        37 => (2, 10.0, 2.0, 2.0, 1, 1, F_DASH | 512 | 128),
        38 => (8, 5.0, 0.0, 0.5, 2, 1, F_DASH | 1024 | F_SCOUT | 512 | F_STIFF),
        39 => (2, 5.0, 0.0, 0.5, 2, 1, F_CARRY | F_DASH | F_FLOAT | 1024 | F_SCOUT | F_STIFF),
        40 => (2, 10.0, 2.0, 2.0, 2, 1, F_CARRY | F_DASH | F_FLOAT | 512 | 128),
        41 => (15, 10.0, 3.0, 2.0, 2, 3, F_CARRY | F_FLOAT | 512 | F_STIFF),
        42 => (5, 10.0, 2.0, 1.0, 3, 2, F_CARRY | F_DASH | F_FLOAT | F_SCOUT | 512),
        43 => (0, 10.0, 0.0, 1.0, 2, 2, F_CARRY | F_FLOAT | 512 | F_STIFF),
        44 => (5, 10.0, 3.0, 3.0, 3, 1, F_CARRY | F_DASH | F_FLOAT | 512),
        45 => (10, 40.0, 4.0, 4.0, 2, 1, F_CARRY | F_FLOAT | 512 | F_STIFF),
        _ => (2, 10.0, 2.0, 2.0, 1, 1, F_DASH),
    }
}

#[derive(Clone)]
struct Act {
    id: String,
    kind: i32,
    src: i32,
    tgt: i32,
    par: i32,
}

struct G {
    size: usize,
    n: usize,
    me: i32,
    terrain: Vec<i32>,
    towner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    road: Vec<bool>,
    capital: Vec<i32>,
    explored: Vec<bool>,
    uowner: Vec<i32>,
    ukind: Vec<i32>,
    uhp: Vec<f32>,
    uvet: Vec<bool>,
    umoved: Vec<bool>,
    uatt: Vec<bool>,
    cowner: Vec<i32>,
    clevel: Vec<i32>,
    cwall: Vec<bool>,
    stars: i32,
    turn: i32,
    tech: Vec<bool>,
    my_units: Vec<usize>,
    my_cities: Vec<usize>,
    foe_units: Vec<usize>,
    foe_cities: Vec<usize>,
    villages: Vec<usize>,
    ruins: Vec<usize>,
    starfish: Vec<usize>,
    acts_this_turn: i32,
    prev_turn: i32,
    f_expand: Vec<i32>,
    f_foe: Vec<i32>,
    f_expl: Vec<i32>,
    f_expand_sea: Vec<i32>,
    f_foe_sea: Vec<i32>,
    f_expl_sea: Vec<i32>,
    navy_needed: bool,
    water_near: bool,
    epot: Vec<i32>,
    prev_tiles: Vec<usize>,
    researched: bool,
    summoned: i32,
}

fn gi(v: &Value, k: &str, d: i64) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(d)
}
fn gb(v: &Value, k: &str, d: bool) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(d)
}
fn gf(v: &Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

impl G {
    fn new() -> G {
        G {
            size: 0,
            n: 0,
            me: 1,
            terrain: Vec::new(),
            towner: Vec::new(),
            structure: Vec::new(),
            resource: Vec::new(),
            road: Vec::new(),
            capital: Vec::new(),
            explored: Vec::new(),
            uowner: Vec::new(),
            ukind: Vec::new(),
            uhp: Vec::new(),
            uvet: Vec::new(),
            umoved: Vec::new(),
            uatt: Vec::new(),
            cowner: Vec::new(),
            clevel: Vec::new(),
            cwall: Vec::new(),
            stars: 0,
            turn: 0,
            tech: vec![false; 128],
            my_units: Vec::new(),
            my_cities: Vec::new(),
            foe_units: Vec::new(),
            foe_cities: Vec::new(),
            villages: Vec::new(),
            ruins: Vec::new(),
            starfish: Vec::new(),
            acts_this_turn: 0,
            prev_turn: -1,
            f_expand: Vec::new(),
            f_foe: Vec::new(),
            f_expl: Vec::new(),
            f_expand_sea: Vec::new(),
            f_foe_sea: Vec::new(),
            f_expl_sea: Vec::new(),
            navy_needed: false,
            water_near: false,
            epot: Vec::new(),
            prev_tiles: Vec::new(),
            researched: false,
            summoned: 0,
        }
    }

    fn parse(&mut self, obs: &Value) {
        let v = &obs["visible_state"];
        self.me = gi(obs, "player", 1) as i32;
        let size = gi(v, "board_size", 14) as usize;
        if size < 2 || size > 64 {
            return;
        }
        self.size = size;
        let n = size * size;
        if self.n != n || self.terrain.len() != n {
            self.n = n;
            let z = || vec![0i32; n];
            self.terrain = z();
            self.towner = z();
            self.structure = z();
            self.resource = z();
            self.road = vec![false; n];
            self.capital = z();
            self.explored = vec![false; n];
            self.uowner = z();
            self.ukind = z();
            self.uhp = vec![0.0; n];
            self.uvet = vec![false; n];
            self.umoved = vec![false; n];
            self.uatt = vec![false; n];
            self.cowner = z();
            self.clevel = z();
            self.cwall = vec![false; n];
            self.tech = vec![false; 128];
            self.epot = vec![0; n];
        }
        for i in 0..n {
            self.terrain[i] = 0;
            self.towner[i] = 0;
            self.structure[i] = 0;
            self.resource[i] = 0;
            self.road[i] = false;
            self.capital[i] = 0;
            self.explored[i] = false;
            self.uowner[i] = 0;
            self.ukind[i] = 0;
            self.uhp[i] = 0.0;
            self.cowner[i] = 0;
            self.clevel[i] = 0;
            self.cwall[i] = false;
        }
        self.stars = gi(v, "stars", 0) as i32;
        let t = gi(v, "turn", 0) as i32;
        if t != self.prev_turn {
            self.prev_turn = t;
            self.acts_this_turn = 0;
            if !self.my_units.is_empty() {
                self.prev_tiles = self.my_units.clone();
            }
            self.researched = false;
            self.summoned = 0;
        }
        self.turn = t;
        if let Some(arr) = v["technology"].as_array() {
            for x in arr {
                let k = x.as_i64().unwrap_or(-1);
                if k >= 0 && k < 128 {
                    self.tech[k as usize] = true;
                }
            }
        }
        if let Some(arr) = v["tiles"].as_array() {
            for row in arr {
                let i = gi(row, "tile", -1);
                if i < 0 || (i as usize) >= n {
                    continue;
                }
                let i = i as usize;
                if !gb(row, "known", false) {
                    continue;
                }
                self.explored[i] = true;
                self.terrain[i] = gi(row, "terrain", 0) as i32;
                self.towner[i] = gi(row, "owner", 0) as i32;
                self.structure[i] = gi(row, "structure", 0) as i32;
                self.resource[i] = gi(row, "resource", 0) as i32;
                self.road[i] = gb(row, "road", false);
                self.capital[i] = gi(row, "capital", 0) as i32;
            }
        }
        if let Some(arr) = v["units"].as_array() {
            for row in arr {
                let i = gi(row, "tile", -1);
                if i < 0 || (i as usize) >= n {
                    continue;
                }
                let i = i as usize;
                self.uowner[i] = gi(row, "owner", 0) as i32;
                self.ukind[i] = gi(row, "kind", 0) as i32;
                self.uhp[i] = gf(row, "health", 0.0) as f32;
                self.uvet[i] = gb(row, "veteran", false);
                self.umoved[i] = gb(row, "moved", false);
                self.uatt[i] = gb(row, "attacked", false);
            }
        }
        if let Some(arr) = v["cities"].as_array() {
            for row in arr {
                let i = gi(row, "tile", -1);
                if i < 0 || (i as usize) >= n {
                    continue;
                }
                let i = i as usize;
                self.cowner[i] = gi(row, "owner", 0) as i32;
                self.clevel[i] = gi(row, "level", 1) as i32;
                if let Some(r) = row["rewards"].as_array() {
                    self.cwall[i] = r.first().and_then(|x| x.as_i64()).unwrap_or(0) > 0;
                }
            }
        }
        self.my_units.clear();
        self.my_cities.clear();
        self.foe_units.clear();
        self.foe_cities.clear();
        self.villages.clear();
        self.ruins.clear();
        self.starfish.clear();
        for i in 0..n {
            if self.uowner[i] == self.me {
                self.my_units.push(i);
            } else if self.uowner[i] > 0 && self.uowner[i] != self.me {
                self.foe_units.push(i);
            }
            if self.structure[i] == 1 {
                if self.cowner[i] == self.me {
                    self.my_cities.push(i);
                } else if self.cowner[i] > 0 {
                    self.foe_cities.push(i);
                } else {
                    self.villages.push(i);
                }
            } else if self.structure[i] == 2 {
                self.ruins.push(i);
            }
            if self.resource[i] == 8 {
                self.starfish.push(i);
            }
        }
        // water proximity and navy need
        self.water_near = false;
        for &c in &self.my_cities {
            for d in 0..self.n {
                if self.cheb(c, d) <= 3 && self.explored[d] && self.terrain[d] <= 2 && self.terrain[d] > 0 {
                    self.water_near = true;
                }
            }
        }
        self.navy_needed = false;
        self.fields();
        self.navy_needed = self.water_near
            && (self.foe_cities.is_empty()
                || self.foe_cities.iter().all(|&c| c < self.f_foe.len() && self.f_foe[c] >= INF));
    }

    fn cheb(&self, a: usize, b: usize) -> i32 {
        let s = self.size as i32;
        let (ax, ay) = ((a % self.size) as i32, (a / self.size) as i32);
        let (bx, by) = ((b % self.size) as i32, (b / self.size) as i32);
        let _ = s;
        (ax - bx).abs().max((ay - by).abs())
    }

    fn nb(&self, i: usize, out: &mut Vec<usize>) {
        out.clear();
        let s = self.size as i32;
        let (x, y) = ((i % self.size) as i32, (i / self.size) as i32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && nx < s && ny >= 0 && ny < s {
                    out.push((ny * s + nx) as usize);
                }
            }
        }
    }

    fn land_pass(&self, i: usize) -> bool {
        let t = self.terrain[i];
        if !self.explored[i] {
            return false;
        }
        if t == 4 {
            return self.tech[20];
        }
        t == 3 || t == 5
    }

    fn bfs_sea(&self, srcs: &[usize]) -> Vec<i32> {
        let n = self.n;
        let mut d = vec![INF; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut nb = Vec::new();
        for &s in srcs {
            if s < n && d[s] == INF {
                d[s] = 0;
                q.push_back(s);
            }
        }
        let ok = |i: usize| self.explored[i] && self.terrain[i] != 0;
        while let Some(x) = q.pop_front() {
            self.nb(x, &mut nb);
            for &y in &nb {
                if d[y] == INF && (y == x || ok(y)) {
                    d[y] = d[x] + 1;
                    q.push_back(y);
                }
            }
        }
        d
    }

    fn bfs(&self, srcs: &[usize]) -> Vec<i32> {
        let n = self.n;
        let mut d = vec![INF; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        let mut nb = Vec::new();
        for &s in srcs {
            if s < n && d[s] == INF {
                d[s] = 0;
                q.push_back(s);
            }
        }
        while let Some(x) = q.pop_front() {
            self.nb(x, &mut nb);
            for &y in &nb {
                if d[y] == INF && (y == x || self.land_pass(y)) {
                    d[y] = d[x] + 1;
                    q.push_back(y);
                }
            }
        }
        d
    }

    fn fields(&mut self) {
        let mut goals: Vec<usize> = Vec::new();
        for &t in self.villages.iter().chain(self.ruins.iter()) {
            goals.push(t);
        }
        for &t in &self.foe_cities {
            goals.push(t);
        }
        self.f_expand = self.bfs(&goals);
        let mut foes: Vec<usize> = Vec::new();
        for &t in &self.foe_units {
            foes.push(t);
        }
        for &t in &self.foe_cities {
            foes.push(t);
        }
        self.f_foe = self.bfs(&foes);
        let mut front: Vec<usize> = Vec::new();
        let mut nb = Vec::new();
        for i in 0..self.n {
            if self.explored[i] {
                self.nb(i, &mut nb);
                if nb.iter().any(|&j| !self.explored[j]) {
                    front.push(i);
                }
            }
        }
        self.f_expl = self.bfs(&front);
        self.f_expand_sea = self.bfs_sea(&goals);
        self.f_foe_sea = self.bfs_sea(&foes);
        self.f_expl_sea = self.bfs_sea(&front);
        if self.epot.len() != self.n {
            self.epot = vec![0; self.n];
        }
        let sz = self.size as i32;
        for i in 0..self.n {
            let (x, y) = ((i % self.size) as i32, (i / self.size) as i32);
            let mut c = 0;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && nx < sz && ny >= 0 && ny < sz {
                        let j = (ny * sz + nx) as usize;
                        if !self.explored[j] {
                            c += 1;
                        }
                    }
                }
            }
            self.epot[i] = c;
        }
    }

    fn maxhp(&self, i: usize) -> f32 {
        ustat(self.ukind[i]).1 + if self.uvet[i] { 5.0 } else { 0.0 }
    }

    fn def_bonus(&self, t: usize) -> f32 {
        let ter = self.terrain[t];
        let foe = self.uowner[t] != self.me;
        let bonus_terrain = if foe { self.turn > 14 } else { false };
        if self.structure[t] == 1 && self.cowner[t] == self.uowner[t] {
            let fort = ustat(self.ukind[t]).6 & F_FORTIFY != 0;
            let mut b: f32 = if fort {
                1.5
            } else {
                1.0
            };
            if !foe && self.cwall[t] {
                b = 4.0;
            }
            if foe && self.turn > 25 {
                b = b.max(2.5);
            }
            b
        } else if ter == 5 && bonus_terrain {
            1.5
        } else if ter == 4 && bonus_terrain {
            1.5
        } else {
            1.0
        }
    }

    fn combat(&self, src: usize, tgt: usize) -> (f32, f32) {
        let (ca, _ha, aa, da, _, ra, fa) = ustat(self.ukind[src]);
        let _ = (ca, da, ra, fa);
        let (cd, hd, _ad, dd, _md, rd, fd) = ustat(self.ukind[tgt]);
        let _ = (cd, hd, _ad, _md, rd, fd);
        let amax = self.maxhp(src);
        let dmax = self.maxhp(tgt);
        let af = aa * (self.uhp[src] / amax.max(1.0));
        let df = dd * (self.uhp[tgt] / dmax.max(1.0)) * self.def_bonus(tgt);
        let tot = af + df;
        if tot <= 0.0 {
            return (0.0, 0.0);
        }
        let dmg = ((af / tot) * aa * 4.5 + 0.5 + 1e-6).floor();
        let dh_after = self.uhp[tgt] - dmg;
        let can_ret = fd & F_STIFF == 0 && self.cheb(src, tgt) <= rd && dh_after > 0.0;
        let ret = if can_ret {
            ((df / tot) * dd * 4.5 + 0.5 + 1e-6).floor()
        } else {
            0.0
        };
        (dmg, ret)
    }

    fn threat_at(&self, dst: usize, kind: i32, hp: f32) -> f32 {
        // rough worst-case damage from visible enemy units next turn
        let (_, _, _, dstat, _, _, _) = ustat(kind);
        let mut worst = 0.0f32;
        for &e in &self.foe_units {
            let (_, _, ea, _, emv, erange, _) = ustat(self.ukind[e]);
            if ea <= 0.0 {
                continue;
            }
            let reach = emv + erange;
            if self.cheb(e, dst) > reach.max(1) {
                continue;
            }
            let af = ea * (self.uhp[e] / self.maxhp(e).max(1.0));
            let df = dstat * (hp / 10.0f32.max(hp)) * self.def_bonus(dst);
            let tot = af + df;
            if tot <= 0.0 {
                continue;
            }
            let dmg = ((af / tot) * ea * 4.5 + 0.5 + 1e-6).floor();
            if dmg > worst {
                worst = dmg;
            }
        }
        worst
    }

    fn decide(&mut self, obs: &Value) -> String {
        self.parse(obs);
        let mut acts: Vec<Act> = Vec::new();
        if let Some(arr) = obs["legal_actions"].as_array() {
            for a in arr {
                let c = &a["command"];
                acts.push(Act {
                    id: a["action_id"].as_str().unwrap_or("").to_string(),
                    kind: gi(c, "kind", 0) as i32,
                    src: gi(c, "source", -1) as i32,
                    tgt: gi(c, "target", -1) as i32,
                    par: gi(c, "parameter", 0) as i32,
                });
            }
        }
        if acts.is_empty() {
            return "0".to_string();
        }
        self.acts_this_turn += 1;
        let at_war = !self.foe_units.is_empty() || !self.foe_cities.is_empty();
        let foe_near_city = self
            .foe_units
            .iter()
            .any(|&e| self.my_cities.iter().any(|&c| self.cheb(c, e) <= 6));
        let best = self.best_action(&acts, at_war, foe_near_city);
        for a in &acts {
            if a.id == best {
                if a.kind == 7 {
                    self.researched = true;
                }
                if a.kind == 4 {
                    self.summoned += 1;
                }
            }
        }
        best
    }

    fn tech_prio(&self, k: i32) -> i32 {
        let metal = (0..self.n).any(|i| self.explored[i] && self.resource[i] == 5 && self.towner[i] == self.me);
        let crop_forest = (0..self.n).any(|i| self.explored[i] && self.terrain[i] == 5 && self.towner[i] == self.me);
        match k {
            6 => 94,  // Organization
            8 => if crop_forest { 100 } else { 92 }, // Farming
            18 => 97, // Archery
            9 => 90,  // Construction
            20 => if metal { 93 } else { 70 }, // Climbing
            23 => if metal { 96 } else { 68 }, // Mining
            24 => if metal { 89 } else { 74 }, // Smithery
            10 => if self.water_near { 95 } else { 60 }, // Fishing
            13 => if self.water_near { 88 } else { 40 }, // Sailing
            44 => if self.water_near { 86 } else { 38 }, // Ramming
            14 => if self.water_near { 66 } else { 30 }, // Navigation
            4 => 80,  // Roads
            5 => 72,  // Trade
            17 => 76, // Mathematics
            16 => 64, // Forestry
            21 => 58, // Meditation
            22 => 56, // Philosophy
            7 => 62,  // Strategy
            3 => 78,  // Chivalry
            2 => 54,  // Free Spirit
            1 => 52,  // Riding
            12 => 34, // Aquatism
            _ => 20,
        }
    }

    fn best_action(&self, acts: &[Act], at_war: bool, foe_near_city: bool) -> String {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_id = String::new();
        let n = self.n;
        let nb = |i: usize, out: &mut Vec<usize>| {
            out.clear();
            let s = self.size as i32;
            let (x, y) = ((i % self.size) as i32, (i / self.size) as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (x + dx, y + dy);
                    if nx >= 0 && nx < s && ny >= 0 && ny < s {
                        out.push((ny * s + nx) as usize);
                    }
                }
            }
        };
        let military_units = self
            .my_units
            .iter()
            .filter(|&&i| ustat(self.ukind[i]).2 > 0.0)
            .count();
        let use_foe = at_war && (military_units >= 2 || foe_near_city || !self.foe_cities.is_empty());
        let my_army: f32 = self.my_units.iter().map(|&i| ustat(self.ukind[i]).2 * self.uhp[i]).sum();
        let foe_army: f32 = self.foe_units.iter().map(|&i| ustat(self.ukind[i]).2 * self.uhp[i]).sum();
        let strong = my_army > foe_army * 1.1 + 4.0;
        let stars = self.stars;
        for a in acts {
            let mut sc: f64 = match a.kind {
                11 => -1e9,
                10 => {
                    if self.acts_this_turn > 60 {
                        1e8
                    } else {
                        10.0
                    }
                }
                9 => {
                    // mandatory city reward
                    let slot = if a.tgt >= 0 && (a.tgt as usize) < n {
                        self.clevel[a.tgt as usize]
                    } else {
                        2
                    };
                    let w = match a.par {
                        3 => 100.0, // workshop: +1 production
                        1 => if at_war { 95.0 } else { 60.0 }, // wall
                        7 => if at_war { 55.0 } else { 90.0 }, // resources +5 stars
                        8 => 92.0,  // population +3
                        5 => if slot >= 4 { 70.0 } else { 88.0 }, // border growth
                        2 => if strong { 60.0 } else { 96.0 }, // park: +1 production
                        6 => if strong || foe_near_city { 97.0 } else { 58.0 }, // giant
                        4 => 40.0,  // explorer
                        _ => 1.0,
                    };
                    1e12 + w as f64
                }
                8 => {
                    let t = a.src as usize;
                    if t < n {
                        if self.structure[t] == 1 && self.cowner[t] > 0 && self.cowner[t] != self.me {
                            1e11
                        } else if self.structure[t] == 1 {
                            9e9 + (100.0 - self.f_expand[t].min(99) as f64)
                        } else if self.structure[t] == 2 {
                            8.5e9
                        } else if self.resource[t] == 8 {
                            8e9
                        } else {
                            7e9
                        }
                    } else {
                        7e9
                    }
                }
                3 => {
                    if a.par == 16 {
                        9.5e9 // promote (free)
                    } else if a.par == 7 {
                        5.0 // recover: end-of-turn healing is free anyway
                    } else if a.par == 1 || a.par == 2 || a.par == 3 {
                        let t = a.tgt as usize;
                        if a.par == 2 {
                            4.2e9 // burn forest: +1 star, opens land
                        } else if a.par == 1 {
                            if stars >= 8 && t < n {
                                6.1e9 // clear forest -> crop, then farm
                            } else {
                                -1e8
                            }
                        } else if stars >= 12 {
                            4.1e9 // grow forest (animal) for harvest
                        } else {
                            -1e8
                        }
                    } else {
                        -1e8
                    }
                }
                2 => {
                    let s = a.src as usize;
                    let t = a.tgt as usize;
                    if s < n && t < n && self.uowner[t] > 0 {
                        let (dmg, ret) = self.combat(s, t);
                        let kills = self.uhp[t] - dmg <= 0.0;
                        let we_die = self.uhp[s] - ret <= 0.0;
                        let mut v = dmg as f64 * 2000.0 - ret as f64 * 2500.0;
                        if kills {
                            v += 4e6 + ustat(self.ukind[t]).0 as f64 * 40000.0;
                            if self.structure[t] == 1 {
                                v += 6e6; // city falls open
                            }
                        }
                        if we_die {
                            v -= 5e6 + ustat(self.ukind[s]).0 as f64 * 40000.0;
                        }
                        if kills && !we_die {
                            v += 5e5;
                        }
                        5e6 + v
                    } else {
                        -1e8
                    }
                }
                7 => {
                    let p = self.tech_prio(a.par);
                    let tier = if [1,6,10,15,20].contains(&a.par) {1} else if [2,4,7,8,13,16,18,21,23,44].contains(&a.par) {2} else {3};
                    let cost = 4 + self.my_cities.len() as i32 * tier;
                    if stars - cost < 2 {
                        -1e8
                    } else if self.researched {
                        4.4e9 + p as f64 * 1e4
                    } else {
                        6e9 + p as f64 * 1e4
                    }
                }
                6 => {
                    let t = a.tgt as usize;
                    let k = a.par;
                    if t >= n {
                        continue;
                    }
                    let (cost, pop) = self.build_value(t, k, &nb);
                    if cost > stars {
                        -1e8
                    } else if pop <= 0.0 && cost > 0 {
                        // roads/bridges/ports: strategic value
                        let mut v = 0.0f64;
                        if k == 8 {
                            v = if self.navy_needed || self.water_near { 6.25e9 } else { 4.3e9 };
                        } else if k == 48 {
                            v = 4.5e9;
                        } else if k == 3 {
                            let useful = self.my_cities.len() >= 2 || self.road_toward(t);
                            v = if useful && stars >= 12 { 4.35e9 } else { -1e8 };
                        } else {
                            v = -1e8;
                        }
                        v
                    } else if stars - cost >= 2 || cost == 0 {
                        6.2e9 + pop as f64 * 1e6 - cost as f64 * 2e4
                    } else {
                        -1e8
                    }
                }
                5 => {
                    if stars >= 4 {
                        6.15e9
                    } else if stars >= 2 {
                        4.95e9
                    } else {
                        -1e8
                    }
                }
                4 => {
                    let k = a.par;
                    let reserve = if foe_near_city || at_war || self.summoned > 0 { 0 } else { 2 };
                    let cost = ustat(k).0;
                    if stars - cost < reserve {
                        -1e8
                    } else {
                        let pr = match k {
                            11 => 97,
                            9 => 96,
                            4 => 94,
                            44 => if self.navy_needed || self.water_near { 93 } else { 50 },
                            8 => 84,
                            42 => if self.navy_needed { 80 } else { 40 },
                            5 => 76,
                            12 => 74,
                            3 => 60,
                            2 => 50,
                            40 => 44,
                            43 => 30,
                            _ => 20,
                        };
                        let undef = self.my_cities.iter().any(|&c| self.uowner[c] == 0) && (foe_near_city || self.turn >= 4);
                        if undef && (k == 5 || k == 9 || k == 11 || k == 2) {
                            6.4e9 + pr as f64 * 1e5
                        } else {
                            4.5e9 + pr as f64 * 1e5
                        }
                    }
                }
                1 => {
                    let s = a.src as usize;
                    let t = a.tgt as usize;
                    if s >= n || t >= n {
                        -1e8
                    } else {
                        self.move_score(s, t, use_foe, &nb)
                    }
                }
                _ => -1e8,
            };
            if a.kind == 1 || a.kind == 2 {
                let _ = &mut sc;
            }
            if sc > best_score {
                best_score = sc;
                best_id = a.id.clone();
            }
        }
        if best_id.is_empty() {
            return acts[0].id.clone();
        }
        best_id
    }

    fn road_toward(&self, t: usize) -> bool {
        if self.my_cities.len() < 2 {
            return false;
        }
        let a = self.my_cities[0];
        let b = self.my_cities[1];
        let d = |x: usize, y: usize| self.cheb(x, y);
        d(a, t) + d(t, b) <= d(a, b) + 2
    }

    fn build_value(&self, t: usize, k: i32, nb: &dyn Fn(usize, &mut Vec<usize>)) -> (i32, f64) {
        let mut adj = Vec::new();
        nb(t, &mut adj);
        let cost = match k {
            5 | 6 | 21 | 22 | 13 | 50 | 48 => 5,
            12 => 3,
            3 => 3,
            8 => 7,
            17 | 19 | 20 | 34 | 35 => 20,
            18 => 15,
            23..=29 => 0,
            _ => 5,
        };
        let pop: f64 = match k {
            5 => 2.0,
            21 => 2.0,
            12 => 1.0,
            6 => adj.iter().filter(|&&j| self.structure[j] == 5 && self.towner[j] == self.me).count() as f64,
            22 => 2.0 * adj.iter().filter(|&&j| self.structure[j] == 21 && self.towner[j] == self.me).count() as f64,
            13 => adj.iter().filter(|&&j| self.structure[j] == 12 && self.towner[j] == self.me).count() as f64,
            50 => {
                let lvl: f64 = adj
                    .iter()
                    .filter(|&&j| (self.structure[j] == 6 || self.structure[j] == 13 || self.structure[j] == 22) && self.towner[j] == self.me)
                    .count() as f64;
                lvl * 3.0
            }
            23..=29 => 3.0,
            17 | 18 | 19 | 20 => 1.0,
            _ => 0.0,
        };
        (cost, pop)
    }

    fn move_score(&self, s: usize, t: usize, use_foe: bool, nb: &dyn Fn(usize, &mut Vec<usize>)) -> f64 {
        let kind = self.ukind[s];
        let (_, _, atk, def, _, range, flags) = ustat(kind);
        let float = flags & F_FLOAT != 0;
        if !float && self.structure[t] == 8 && self.towner[t] == self.me && self.terrain[t] <= 2 {
            return if self.navy_needed || (self.water_near && self.turn > 10) {
                3e6
            } else {
                -1e8
            };
        }
        let field = if float {
            if use_foe && atk > 0.0 && self.f_foe_sea[s] < INF {
                &self.f_foe_sea
            } else if self.f_expand_sea[s] < INF || self.f_expand_sea[t] < INF {
                &self.f_expand_sea
            } else {
                &self.f_expl_sea
            }
        } else if use_foe && atk > 0.0 && self.f_foe[s] < INF {
            &self.f_foe
        } else if self.f_expand[s] < INF || self.f_expand[t] < INF {
            &self.f_expand
        } else {
            &self.f_expl
        };
        let prog = field[s] - field[t];
        let mut nbv = Vec::new();
        nb(t, &mut nbv);
        let mut unseen = 0;
        for &j in nbv.iter() {
            if !self.explored[j] {
                unseen += 1;
            }
        }
        let mut enable = false;
        let mut enable_v = 0.0f64;
        if atk > 0.0 && flags & F_DASH != 0 {
            for &e in &self.foe_units {
                if self.cheb(t, e) <= range {
                    let (dmg, ret) = self.combat_at(s, t, e);
                    let kills = self.uhp[e] - dmg <= 0.0;
                    let we_die = self.uhp[s] - ret <= 0.0;
                    if kills && !we_die {
                        enable = true;
                        enable_v = 3e6;
                    } else if dmg > ret && !we_die {
                        enable = true;
                        enable_v = (enable_v.max(0.0)).max(6e5);
                    }
                }
            }
        }
        let ep = self.epot[t] - self.epot[s];
        // only move when it achieves something
        if prog <= 0 && unseen == 0 && !enable && ep <= 0 {
            return -1e8;
        }
        let threat = self.threat_at(t, kind, self.uhp[s]);
        let hurt = self.uhp[s] < self.maxhp(s);
        let mut sc = 1e6 + prog.max(0) as f64 * 30000.0 + prog.min(0) as f64 * 120000.0 + enable_v;
        sc += unseen as f64 * 8000.0 + ep.max(0) as f64 * 9000.0;
        if self.prev_tiles.contains(&t) && !enable {
            sc -= 900000.0;
        }
        sc -= threat as f64 * 25000.0;
        if threat >= self.uhp[s] {
            sc -= 4e6;
        }
        let ter = self.terrain[t];
        let defbonus = if ter == 5 { 1.5 } else if ter == 4 { 1.5 } else if self.structure[t] == 1 && self.cowner[t] == self.me { 1.5 } else { 1.0 };
        if threat > 0.0 || hurt {
            sc += (defbonus - 1.0) * 60000.0;
        } else {
            sc += (defbonus - 1.0) * 8000.0;
        }
        // do not strip defenders from a city when the enemy is close
        let _ = def;
        if self.structure[s] == 1 && self.cowner[s] == self.me && self.foe_units.iter().any(|&e| self.cheb(e, s) <= 7) {
            sc -= 500000.0;
        }
        sc
    }

    fn combat_at(&self, src: usize, from: usize, tgt: usize) -> (f32, f32) {
        let (_, _, aa, _, _, _, _) = ustat(self.ukind[src]);
        let (_, _, _, dd, _, rd, fd) = ustat(self.ukind[tgt]);
        let amax = self.maxhp(src);
        let dmax = self.maxhp(tgt);
        let af = aa * (self.uhp[src] / amax.max(1.0));
        let df = dd * (self.uhp[tgt] / dmax.max(1.0)) * self.def_bonus(tgt);
        let tot = af + df;
        if tot <= 0.0 {
            return (0.0, 0.0);
        }
        let dmg = ((af / tot) * aa * 4.5 + 0.5 + 1e-6).floor();
        let can_ret = fd & F_STIFF == 0 && self.cheb(from, tgt) <= rd && self.uhp[tgt] - dmg > 0.0;
        let ret = if can_ret {
            ((df / tot) * dd * 4.5 + 0.5 + 1e-6).floor()
        } else {
            0.0
        };
        (dmg, ret)
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut g = G::new();
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
        let id = g.decide(&obs);
        let _ = writeln!(stdout, "{{\"action_id\":\"{}\"}}", id);
        let _ = stdout.flush();
    }
}
