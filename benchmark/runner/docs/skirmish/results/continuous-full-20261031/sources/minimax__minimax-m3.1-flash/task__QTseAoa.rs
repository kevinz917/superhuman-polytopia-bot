// Polytopia Bardur 1v1 bot: economy-first expansion with a knight army.
//
// One JSON observation per line on stdin; reply {"action_id":"..."} and flush.
use serde_json::Value;
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

// ---------------------------------------------------------------- rule tables

fn u_stats(kind: i32) -> (f32, f32, f32, i32, i32, i32) {
    // hp, attack, defense, movement, range, cost
    match kind {
        2 => (10.0, 2.0, 2.0, 1, 1, 2),    // warrior
        3 => (10.0, 2.0, 1.0, 2, 1, 3),    // rider
        4 => (10.0, 3.5, 1.0, 3, 1, 8),    // knight
        5 => (15.0, 1.0, 3.0, 1, 1, 3),    // defender
        8 => (10.0, 4.0, 0.0, 1, 3, 8),    // catapult
        9 => (10.0, 2.0, 1.0, 1, 2, 3),    // archer
        10 => (10.0, 0.0, 1.0, 1, 1, 5),   // mind bender
        11 => (15.0, 3.0, 3.0, 1, 1, 5),   // swordsman
        12 => (40.0, 5.0, 4.0, 1, 1, 10),  // giant
        37 => (10.0, 2.0, 2.0, 1, 1, 2),   // dagger
        38 => (5.0, 0.0, 0.5, 2, 1, 8),    // cloak
        39 => (5.0, 0.0, 0.5, 2, 1, 2),    // dinghy
        40 => (10.0, 2.0, 2.0, 2, 1, 2),   // pirate
        41 => (10.0, 3.0, 2.0, 2, 3, 15),  // bomber
        42 => (10.0, 2.0, 1.0, 3, 2, 5),   // scoutship
        43 => (10.0, 0.0, 1.0, 2, 2, 0),   // raft
        44 => (10.0, 3.0, 3.0, 3, 1, 5),   // rammership
        45 => (40.0, 4.0, 4.0, 2, 1, 10),  // juggernaut
        _ => (10.0, 1.0, 1.0, 1, 1, 3),
    }
}

fn u_stiff(kind: i32) -> bool {
    matches!(kind, 8 | 10 | 22 | 34 | 38 | 39 | 41 | 43 | 60 | 61 | 62)
}

fn u_surprise(kind: i32) -> bool {
    matches!(kind, 30 | 37 | 40 | 61)
}

fn u_fortify(kind: i32) -> bool {
    matches!(kind, 2 | 3 | 4 | 5 | 9 | 15 | 17 | 25 | 37 | 38 | 39 | 40)
}

fn u_escape(kind: i32) -> bool {
    matches!(kind, 3 | 18 | 20 | 23 | 24 | 26 | 28 | 29 | 35 | 44)
}

fn u_float(kind: i32) -> bool {
    matches!(kind, 22 | 26 | 27 | 35 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 57)
}

// tech id -> (tier, required prereq)
const PLAN: &[(i32, i32, i32)] = &[
    (1, 1, -1),   // riding
    (6, 1, -1),   // organization
    (2, 2, 1),    // free spirit
    (3, 3, 2),    // chivalry (knights)
    (10, 1, -1),  // fishing (ports, rafts)
    (18, 2, 15),  // archery
    (8, 2, 6),    // farming
    (9, 3, 8),    // construction (windmill)
    (7, 2, 6),    // strategy (defenders)
    (4, 2, 1),    // roads
    (5, 3, 4),    // trade (market)
    (20, 1, -1),  // climbing
    (23, 2, 20),  // mining
    (24, 3, 23),  // smithery (swordsman)
    (21, 2, 20),  // meditation
    (22, 3, 21),  // philosophy (mind bender)
];

// ---------------------------------------------------------------- observation

#[derive(Clone, Copy, Default)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    resource: i32,
    structure: i32,
    capital: i32,
}

#[derive(Clone, Copy)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    hp: f32,
    veteran: bool,
    moved: bool,
    attacked: bool,
    kills: i32,
    home: i32,
    passenger: i32,
}

#[derive(Clone)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    progress: i32,
    population: i32,
    production: i32,
    wall: bool,
}

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
    other: i32,
    tech: Vec<bool>,
    tiles: Vec<Tile>,
    units: Vec<Unit>,
    cities: Vec<City>,
    acts: Vec<Act>,
    met: bool,
}

fn gi(v: &Value, k: &str) -> i32 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(-1) as i32
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}
fn gf(v: &Value, k: &str) -> f32 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
}

fn parse(v: &Value) -> Obs {
    let vs = &v["visible_state"];
    let size = vs["board_size"].as_i64().unwrap_or(14) as usize;
    let n = size * size;
    let me = v["player"].as_i64().unwrap_or(1) as i32;
    let other = if me == 1 { 2 } else { 1 };
    let mut tiles = vec![Tile::default(); n];
    if let Some(arr) = vs["tiles"].as_array() {
        for t in arr {
            let i = gi(t, "tile");
            if i < 0 || i as usize >= n {
                continue;
            }
            let known = gb(t, "known");
            tiles[i as usize] = Tile {
                known,
                terrain: if known { gi(t, "terrain") } else { -1 },
                owner: if known { gi(t, "owner") } else { -1 },
                resource: if known { gi(t, "resource") } else { -1 },
                structure: if known { gi(t, "structure") } else { -1 },
                capital: if known { gi(t, "capital") } else { -1 },
            };
        }
    }
    let mut units = Vec::new();
    if let Some(arr) = vs["units"].as_array() {
        for u in arr {
            let i = gi(u, "tile");
            if i < 0 || i as usize >= n {
                continue;
            }
            units.push(Unit {
                tile: i as usize,
                owner: gi(u, "owner"),
                kind: gi(u, "kind"),
                hp: gf(u, "health"),
                veteran: gb(u, "veteran"),
                moved: gb(u, "moved"),
                attacked: gb(u, "attacked"),
                kills: gi(u, "kills"),
                home: gi(u, "home"),
                passenger: gi(u, "passenger"),
            });
        }
    }
    let mut cities = Vec::new();
    if let Some(arr) = vs["cities"].as_array() {
        for c in arr {
            let i = gi(c, "tile");
            if i < 0 || i as usize >= n {
                continue;
            }
            let mut wall = false;
            if let Some(r) = c["rewards"].as_array() {
                if r.len() > 1 {
                    wall = r[1].as_i64().unwrap_or(0) > 0;
                }
            }
            cities.push(City {
                tile: i as usize,
                owner: gi(c, "owner"),
                level: gi(c, "level"),
                progress: gi(c, "progress"),
                population: gi(c, "population"),
                production: gi(c, "production"),
                wall,
            });
        }
    }
    let mut tech = vec![false; 128];
    if let Some(arr) = vs["technology"].as_array() {
        for t in arr {
            let i = t.as_i64().unwrap_or(-1);
            if i >= 0 && i < 128 {
                tech[i as usize] = true;
            }
        }
    }
    let mut acts = Vec::new();
    if let Some(arr) = v["legal_actions"].as_array() {
        for a in arr {
            let c = &a["command"];
            acts.push(Act {
                id: a["action_id"].as_str().unwrap_or("").to_string(),
                kind: gi(c, "kind"),
                src: gi(c, "source"),
                tgt: gi(c, "target"),
                par: gi(c, "parameter"),
            });
        }
    }
    let mut met = false;
    if let Some(arr) = vs["relations"]["known_players"].as_array() {
        let idx = (other - 1) as usize;
        if idx < arr.len() {
            met = arr[idx].as_bool().unwrap_or(false);
        }
    }
    Obs {
        size,
        n,
        turn: vs["turn"].as_i64().unwrap_or(0) as i32,
        stars: vs["stars"].as_i64().unwrap_or(0) as i32,
        me,
        other,
        tech,
        tiles,
        units,
        cities,
        acts,
        met,
    }
}

// ---------------------------------------------------------------- helpers

fn cheb(size: usize, a: usize, b: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

fn adjacent(size: usize, a: usize, b: usize) -> bool {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs() <= 1 && (ay - by).abs() <= 1
}

fn near(size: usize, a: usize, b: usize, r: i32) -> bool {
    cheb(size, a, b) <= r
}

// ---------------------------------------------------------------- bot

struct Bot {
    capital: Option<usize>,
    eunits: HashMap<usize, (i32, f32, i32)>,
    ecities: HashMap<usize, i32>,
    turn: i32,
}

impl Bot {
    fn new() -> Bot {
        Bot { capital: None, eunits: HashMap::new(), ecities: HashMap::new(), turn: 0 }
    }

    fn memory(&mut self, o: &Obs) {
        self.turn = o.turn;
        for t in &o.tiles {
            if t.known && t.capital == o.me && self.capital.is_none() {
                self.capital = Some(t.terrain as usize); // placeholder, fixed below
            }
        }
        // capital: tile whose capital field equals my seat
        for (i, t) in o.tiles.iter().enumerate() {
            if t.known && t.capital == o.me {
                self.capital = Some(i);
                break;
            }
        }
        for u in &o.units {
            if u.owner == o.other {
                self.eunits.insert(u.tile, (u.kind, u.hp, o.turn));
            }
        }
        for c in &o.cities {
            if c.owner == o.other {
                self.ecities.insert(c.tile, o.turn);
            }
        }
        // forget stale sightings
        let old = o.turn - 12;
        self.eunits.retain(|_, (_, _, t)| *t >= old);
        self.ecities.retain(|_, t| *t >= old);
    }

    fn my_cities<'a>(&self, o: &'a Obs) -> Vec<&'a City> {
        o.cities.iter().filter(|c| c.owner == o.me).collect()
    }

    fn city_count(&self, o: &Obs) -> i32 {
        o.cities.iter().filter(|c| c.owner == o.me).count() as i32
    }

    fn next_tech(&self, o: &Obs) -> Option<(i32, i32, i32)> {
        for &(t, tier, pre) in PLAN {
            if !o.tech[t as usize] && (pre < 0 || o.tech[pre as usize]) {
                let cost = 4 + self.city_count(o) * tier;
                return Some((t, tier, cost));
            }
        }
        None
    }

    // ---- combat -------------------------------------------------------

    fn max_hp(&self, o: &Obs, u: &Unit) -> f32 {
        let kind = if u.passenger > 0 { u.passenger } else { u.kind };
        let (hp, _, _, _, _, _) = u_stats(kind);
        hp + if u.veteran { 5.0 } else { 0.0 }
    }

    /// Defense bonus of the unit standing on `tile` (defender's owner techs).
    fn def_bonus(&self, o: &Obs, tile: usize, owner: i32, kind: i32) -> f32 {
        let t = &o.tiles[tile];
        let tech = &o.tech; // only my techs are known; approximate with mine
        let mut bonus;
        match t.terrain {
            1 | 2 => bonus = if tech[12] { 1.5 } else { 1.0 },
            5 => bonus = if tech[18] { 1.5 } else { 1.0 },
            4 => bonus = if tech[20] { 1.5 } else { 1.0 },
            _ => {
                let mut b = 1.0;
                if u_fortify(kind) {
                    if let Some(c) = o.cities.iter().find(|c| c.tile == tile) {
                        if c.owner == owner {
                            b = if c.wall { 4.0 } else { 1.5 };
                        }
                    }
                }
                bonus = b;
            }
        }
        bonus
    }

    /// Expected damage of `src` (attacker) against `tgt` (defender).
    fn combat(&self, o: &Obs, a: &Unit, d: &Unit) -> (f32, f32, bool) {
        let (_, aatk, _, _, _, _) = u_stats(a.kind);
        let (_, _, ddef, _, drange, _) = u_stats(d.kind);
        let ahp = self.max_hp(o, a);
        let dhp = self.max_hp(o, d);
        let db = self.def_bonus(o, d.tile, d.owner, d.kind);
        let af = aatk * (a.hp / ahp.max(1.0));
        let df = ddef * (d.hp / dhp.max(1.0)) * db;
        let total = af + df;
        let div = if total > 0.0 { total } else { 1.0 };
        let dmg = (((af / div) * aatk * 4.5) + 0.000001 + 0.5).floor();
        let ret = if d.hp > dmg && total > 0.0 {
            (((df / div) * ddef * 4.5) + 0.000001 + 0.5).floor()
        } else {
            0.0
        };
        let retaliates = !u_surprise(a.kind)
            && !u_stiff(d.kind)
            && cheb(o.size, a.tile, d.tile) <= drange
            && o.tiles[a.tile].known;
        (dmg, if retaliates { ret } else { 0.0 }, dmg >= d.hp)
    }

    /// Score of an attack action; None when it should not be taken.
    fn eval_attack(&self, o: &Obs, a: &Act) -> Option<f64> {
        let atk = o.units.iter().find(|u| u.tile == a.src as usize)?;
        let dfn = o.units.iter().find(|u| u.tile == a.tgt as usize)?;
        if dfn.owner != o.other {
            return None;
        }
        let (dmg, ret, kill) = self.combat(o, atk, dfn);
        let my_hp = self.max_hp(o, atk);
        let mut s = 0.0;
        if kill {
            s += 60.0;
            let (_, _, _, _, _, cost) = u_stats(dfn.kind);
            s += (cost as f64) * 4.0;
        } else {
            // trade: only worth it if we clearly win the exchange
            s += (dmg * 2.0 - ret * 3.0) as f64;
            if dmg <= 0.0 {
                return None;
            }
            if ret >= my_hp {
                return None; // we would die
            }
        }
        // value of removing a unit near our cities
        for c in o.cities.iter().filter(|c| c.owner == o.me) {
            let d = cheb(o.size, dfn.tile, c.tile);
            if d <= 2 {
                s += 25.0 - 6.0 * d as f64;
            }
        }
        // a defender in a walled city is hard to kill: do not trade into it
        if !kill {
            let mut in_city = false;
            for c in o.cities.iter() {
                if c.tile == dfn.tile && c.owner == o.other {
                    in_city = true;
                }
            }
            if in_city && dmg * 2.0 < dfn.hp {
                return None;
            }
        }
        if s <= 0.0 {
            return None;
        }
        Some(s)
    }

    // ---- targets ------------------------------------------------------

    /// Nearest capturable village (neutral first, then enemy) to `from`.
    fn village_target(&self, o: &Obs, from: usize) -> Option<usize> {
        let mut best: Option<(i32, usize)> = None;
        for (i, t) in o.tiles.iter().enumerate() {
            if !t.known || t.structure != 1 || t.owner == o.me {
                continue;
            }
            let prio = if t.owner == 0 { 0 } else { 1 };
            let d = cheb(o.size, from, i);
            let key = prio * 1000 + d;
            if best.is_none() || key < best.unwrap().0 {
                best = Some((key, i));
            }
        }
        best.map(|(_, t)| t)
    }

    /// Known tile adjacent to an unknown tile, nearest to `from` (or farthest for advances).
    fn frontier(&self, o: &Obs, from: usize, far: bool) -> Option<usize> {
        let mut best: Option<(i32, usize)> = None;
        for (i, t) in o.tiles.iter().enumerate() {
            if !t.known {
                continue;
            }
            let mut adj_unknown = false;
            let (x, y) = ((i % o.size) as i32, (i / o.size) as i32);
            for dy in -1..=1i32 {
                for dx in -1..=1i32 {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= o.size as i32 || ny >= o.size as i32 {
                        continue;
                    }
                    let j = (ny * o.size as i32 + nx) as usize;
                    if !o.tiles[j].known {
                        adj_unknown = true;
                    }
                }
            }
            if !adj_unknown {
                continue;
            }
            let d = cheb(o.size, from, i);
            let key = if far { -d } else { d };
            if best.is_none() || key < best.unwrap().0 {
                best = Some((key, i));
            }
        }
        best.map(|(_, t)| t)
    }

    fn enemy_city_target(&self, o: &Obs, from: usize) -> Option<usize> {
        let mut best: Option<(i32, usize)> = None;
        for (tile, last) in self.ecities.iter() {
            if *last < o.turn - 6 {
                continue;
            }
            let d = cheb(o.size, from, *tile);
            if best.is_none() || d < best.unwrap().0 {
                best = Some((d, *tile));
            }
        }
        best.map(|(_, t)| t)
    }

    fn enemy_unit_target(&self, o: &Obs, from: usize) -> Option<usize> {
        let mut best: Option<(i32, usize)> = None;
        for (tile, (kind, _, last)) in self.eunits.iter() {
            if *last < o.turn - 6 {
                continue;
            }
            let d = cheb(o.size, from, *tile);
            let mut key = d;
            if u_float(*kind) {
                key += 6; // boats are not the priority
            }
            if best.is_none() || key < best.unwrap().0 {
                best = Some((key, *tile));
            }
        }
        best.map(|(_, t)| t)
    }

    // ---- main ---------------------------------------------------------

    fn choose(&mut self, o: &Obs) -> String {
        if o.acts.is_empty() {
            return String::new();
        }
        self.memory(o);

        // 1. mandatory city reward
        if let Some(id) = self.pick_reward(o) {
            return id;
        }
        // 2. attacks that clearly win
        if let Some(id) = self.pick_attack(o) {
            return id;
        }
        // 3. captures (free territory and income)
        if let Some(id) = self.pick_capture(o) {
            return id;
        }
        // 4. a harvest that pushes a city to its next level
        if let Some(id) = self.pick_harvest(o, true) {
            return id;
        }
        // 5. research
        if let Some(id) = self.pick_research(o) {
            return id;
        }
        // 6. recruitment
        if let Some(id) = self.pick_summon(o) {
            return id;
        }
        // 7. surplus harvesting for city growth
        if let Some(id) = self.pick_harvest(o, false) {
            return id;
        }
        // 7. build
        if let Some(id) = self.pick_build(o) {
            return id;
        }
        // 8. move
        if let Some(id) = self.pick_move(o) {
            return id;
        }
        // 9. end turn
        for a in &o.acts {
            if a.kind == 10 {
                return a.id.clone();
            }
        }
        o.acts[0].id.clone()
    }

    fn pick_reward(&self, o: &Obs) -> Option<String> {
        let mut best: Option<(f64, String)> = None;
        for a in &o.acts {
            if a.kind != 9 {
                continue;
            }
            let city = o.cities.iter().find(|c| c.tile == a.tgt as usize);
            let level = city.map(|c| c.level).unwrap_or(1);
            let mut s = 0.0;
            match a.par {
                4 => {
                    // explorer: reveals a good chunk of the map
                    s = 40.0;
                    let mut villages = 0;
                    for t in &o.tiles {
                        if t.known && t.structure == 1 && t.owner != o.me {
                            villages += 1;
                        }
                    }
                    if villages > 4 {
                        s = 10.0;
                    }
                }
                3 => s = 35.0,  // workshop: +1 production
                1 => {
                    // wall: strong defence
                    s = 30.0;
                    let mut near = false;
                    for u in o.units.iter().filter(|u| u.owner == o.other) {
                        if cheb(o.size, u.tile, a.tgt as usize) <= 4 {
                            near = true;
                        }
                    }
                    if !near {
                        s = 5.0;
                    }
                }
                7 => s = 45.0, // resources +5 stars
                5 => s = 30.0, // border growth
                8 => s = 35.0, // population +3
                6 => s = 40.0, // giant
                2 => s = 30.0, // park
                _ => s = 0.0,
            }
            let _ = level;
            if best.is_none() || s > best.as_ref().unwrap().0 {
                best = Some((s, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn pick_attack(&self, o: &Obs) -> Option<String> {
        let mut best: Option<(f64, String)> = None;
        for a in &o.acts {
            if a.kind != 2 {
                continue;
            }
            if let Some(s) = self.eval_attack(o, a) {
                if best.is_none() || s > best.as_ref().unwrap().0 {
                    best = Some((s, a.id.clone()));
                }
            }
        }
        best.map(|(_, id)| id)
    }

    fn pick_capture(&self, o: &Obs) -> Option<String> {
        let mut best: Option<(f64, String)> = None;
        for a in &o.acts {
            if a.kind != 8 {
                continue;
            }
            let t = &o.tiles[a.src as usize];
            let mut s = 10.0;
            if t.structure == 1 && t.owner == o.other {
                // enemy city: worth a lot, but check it is not a walled stronghold
                s = 80.0;
                let mut def = 0;
                for u in o.units.iter() {
                    if u.owner == o.other && near(o.size, u.tile, a.src as usize, 1) {
                        def += 1;
                    }
                }
                s -= (def as f64) * 20.0;
            } else if t.structure == 1 {
                s = 25.0; // neutral village
            } else if t.structure == 2 {
                s = 12.0; // ruin
            } else {
                s = 5.0; // starfish
            }
            if s <= 0.0 {
                continue;
            }
            if best.is_none() || s > best.as_ref().unwrap().0 {
                best = Some((s, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn pick_research(&self, o: &Obs) -> Option<String> {
        // cheapest first-in-plan that is legal
        let mut best: Option<(usize, String)> = None;
        for (i, a) in o.acts.iter().enumerate() {
            if a.kind != 7 {
                continue;
            }
            let rank = PLAN.iter().position(|&(t, _, _)| t == a.par);
            let rank = match rank {
                Some(r) => r,
                None => continue,
            };
            if best.is_none() || rank < best.as_ref().unwrap().0 {
                best = Some((rank, a.id.clone()));
            }
            let _ = i;
        }
        best.map(|(_, id)| id)
    }

    fn reserve(&self, o: &Obs) -> i32 {
        match self.next_tech(o) {
            Some((_, _, cost)) => cost.min(24),
            None => 0,
        }
    }

    /// How many military units we want right now.
    fn army_target(&self, o: &Obs) -> (usize, usize) {
        let mut warriors = 0;
        for u in o.units.iter().filter(|u| u.owner == o.me) {
            if u.kind == 2 || u.kind == 3 || u.kind == 37 {
                warriors += 1;
            }
        }
        let expl = if o.met { 1 } else { 3 };
        let mut army = 3 + (o.turn / 7) as usize;
        if army > 11 {
            army = 11;
        }
        if o.met && army < 6 {
            army = 6;
        }
        (warriors, expl.min(army))
    }

    /// True when more units would help (used to decide to step the capital defender aside).
    fn want_recruit(&self, o: &Obs) -> bool {
        let mut army = 3 + (o.turn / 7) as usize;
        if army > 11 {
            army = 11;
        }
        if o.met && army < 6 {
            army = 6;
        }
        let mut mine = 0;
        for u in o.units.iter().filter(|u| u.owner == o.me) {
            mine += 1;
        }
        if mine >= army {
            return false;
        }
        // is there room in any of my cities?
        let mut room = false;
        for c in o.cities.iter().filter(|c| c.owner == o.me) {
            let mut used = 0;
            for u in o.units.iter().filter(|u| u.owner == o.me) {
                if u.home == c.tile as i32 + 1 {
                    used += 1;
                }
            }
            if used < c.level {
                room = true;
            }
        }
        room && o.stars >= 3
    }

    /// True when the next planned technology is close enough to be worth saving for.
    fn saving_for_tech(&self, o: &Obs) -> bool {
        match self.next_tech(o) {
            Some((_, _, cost)) => {
                let need = cost - o.stars;
                need > 0 && need <= 4
            }
            None => false,
        }
    }

    fn pick_summon(&self, o: &Obs) -> Option<String> {
        let mut mine0 = 0;
        for u in o.units.iter().filter(|u| u.owner == o.me) {
            mine0 += 1;
        }
        if self.saving_for_tech(o) && mine0 >= 2 {
            return None;
        }
        let (warriors, expl) = self.army_target(o);
        let mut mine = 0;
        for u in o.units.iter().filter(|u| u.owner == o.me) {
            mine += 1;
        }
        let mut prefs: Vec<i32> = Vec::new();
        if o.tech[3] && o.stars >= 10 && mine >= 2 {
            prefs.push(4); // knight
        }
        if warriors < expl {
            if o.tech[1] {
                prefs.push(3); // rider
            }
            prefs.push(2); // warrior
        }
        if o.tech[18] && o.stars >= 7 && mine < 12 {
            prefs.push(9); // archer
        }
        if o.tech[7] && o.stars >= 7 && mine < 12 {
            prefs.push(5); // defender
        }
        if o.tech[3] && o.stars >= 11 {
            prefs.push(4); // knight
        }
        for p in prefs {
            for a in &o.acts {
                if a.kind == 4 && a.par == p {
                    return Some(a.id.clone());
                }
            }
        }
        None
    }

    fn pick_harvest(&self, o: &Obs, level_up_only: bool) -> Option<String> {
        self.pick_harvest2(o, level_up_only)
    }

    fn pick_harvest2(&self, o: &Obs, level_up_only: bool) -> Option<String> {
        let mut lvl: Option<(f64, String)> = None;
        let mut growth: Option<(f64, String)> = None;
        for a in &o.acts {
            if a.kind != 5 {
                continue;
            }
            let t = a.tgt as usize;
            let mut bestcity: Option<&City> = None;
            for c in o.cities.iter().filter(|c| c.owner == o.me) {
                if bestcity.is_none()
                    || cheb(o.size, c.tile, t) < cheb(o.size, bestcity.unwrap().tile, t)
                {
                    bestcity = Some(c);
                }
            }
            let mut s = 1.0;
            let mut level_up = false;
            let mut early = false;
            if let Some(c) = bestcity {
                if c.progress + 1 >= c.level + 1 {
                    level_up = true;
                }
                if c.level < 3 && c.progress + 1 >= c.level {
                    early = true;
                }
                s += 10.0 / (c.level as f64 + 1.0);
            }
            if level_up && o.stars >= 2 {
                if lvl.is_none() || s > lvl.as_ref().unwrap().0 {
                    lvl = Some((s, a.id.clone()));
                }
            } else if !level_up_only && early && o.stars >= 3 {
                if growth.is_none() || s > growth.as_ref().unwrap().0 {
                    growth = Some((s, a.id.clone()));
                }
            }
        }
        if level_up_only {
            return lvl.map(|(_, id)| id);
        }
        if let Some((_, id)) = lvl {
            return Some(id);
        }
        growth.map(|(_, id)| id)
    }

    fn pick_build(&self, o: &Obs) -> Option<String> {
        let res = self.reserve(o);
        if o.stars < res + 12 {
            return None;
        }
        let mut best: Option<(f64, String)> = None;
        for a in &o.acts {
            if a.kind != 6 {
                continue;
            }
            let t = a.tgt as usize;
            let mut s = 0.0;
            match a.par {
                5 => {
                    // farm on a crop tile
                    s = 22.0;
                }
                6 => {
                    s = 20.0;
                }
                8 => {
                    // port
                    s = 18.0;
                }
                3 => {
                    s = 6.0;
                }
                50 => {
                    s = 25.0;
                }
                _ => s = 0.0,
            }
            let _ = t;
            if s > 0.0 && (best.is_none() || s > best.as_ref().unwrap().0) {
                best = Some((s, a.id.clone()));
            }
        }
        best.map(|(_, id)| id)
    }

    fn pick_move(&self, o: &Obs) -> Option<String> {
        let cap = self.capital;
        let mut idle: Vec<&Unit> = o
            .units
            .iter()
            .filter(|u| u.owner == o.me && !u.moved && !u.attacked)
            .collect();
        if idle.is_empty() {
            return None;
        }
        idle.sort_by_key(|u| (u.kind != 2 && u.kind != 3, u.tile));
        let mine = o.units.iter().filter(|u| u.owner == o.me).count();
        let (_, expl) = self.army_target(o);
        let n = idle.len();
        // roles: 0 step aside, 1 go home, 2 explore, 3 army, 9 hold
        let mut role = vec![3u8; n];
        let step_aside = self.want_recruit(o);
        if let Some(c) = cap {
            if let Some(ci) = idle.iter().position(|u| u.tile == c) {
                role[ci] = if step_aside { 0 } else { 9 };
            }
            let mut near = 0;
            for u in o.units.iter().filter(|u| u.owner == o.me) {
                if cheb(o.size, u.tile, c) <= 2 {
                    near += 1;
                }
            }
            if near == 0 {
                let mut best: Option<(i32, usize)> = None;
                for (i, u) in idle.iter().enumerate() {
                    if role[i] == 9 {
                        continue;
                    }
                    let d = cheb(o.size, u.tile, c);
                    if best.is_none() || d < best.unwrap().0 {
                        best = Some((d, i));
                    }
                }
                if let Some((_, i)) = best {
                    role[i] = 1;
                }
            }
        }
        let mut ec = 0;
        for i in 0..n {
            if role[i] != 3 {
                continue;
            }
            let k = idle[i].kind;
            if ec < expl && (k == 2 || k == 3 || k == 37) {
                role[i] = 2;
                ec += 1;
            }
        }
        for want_role in [0u8, 1, 2, 3] {
            let mut best: Option<(f64, String)> = None;
            for i in 0..n {
                if role[i] != want_role {
                    continue;
                }
                let u = idle[i];
                let target: Option<usize> = match role[i] {
                    0 | 1 => cap,
                    2 => self
                        .village_target(o, u.tile)
                        .or_else(|| self.frontier(o, u.tile, false)),
                    _ => self
                        .enemy_city_target(o, u.tile)
                        .or_else(|| self.enemy_unit_target(o, u.tile))
                        .or_else(|| {
                            if mine >= 4 {
                                self.frontier(o, u.tile, true)
                            } else {
                                None
                            }
                        }),
                };
                let target = match target {
                    Some(t) => t,
                    None => continue,
                };
                for a in &o.acts {
                    if a.kind != 1 || a.src != u.tile as i32 {
                        continue;
                    }
                    let dt = a.tgt as usize;
                    let d = cheb(o.size, dt, target);
                    let mut s = -(d as f64) * 10.0;
                    let tt = &o.tiles[dt];
                    if role[i] == 2 {
                        let (x, y) = ((dt % o.size) as i32, (dt / o.size) as i32);
                        for dy in -1..=1i32 {
                            for dx in -1..=1i32 {
                                let nx = x + dx;
                                let ny = y + dy;
                                if nx < 0 || ny < 0 || nx >= o.size as i32 || ny >= o.size as i32 {
                                    continue;
                                }
                                let j = (ny * o.size as i32 + nx) as usize;
                                if !o.tiles[j].known {
                                    s += 0.3;
                                }
                            }
                        }
                    }
                    // do not stand next to a visible enemy
                    for e in o.units.iter().filter(|u| u.owner == o.other) {
                        if cheb(o.size, dt, e.tile) <= 1 {
                            s -= 12.0;
                        }
                    }
                    if tt.owner == o.me && tt.structure == 1 && role[i] != 0 {
                        s -= 0.6;
                    }
                    if best.is_none() || s > best.as_ref().unwrap().0 {
                        best = Some((s, a.id.clone()));
                    }
                }
            }
            if let Some((_, id)) = best {
                return Some(id);
            }
        }
        None
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
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
        let o = parse(&v);
        let id = bot.choose(&o);
        let _ = writeln!(stdout, "{{\"action_id\":\"{}\"}}", id);
        let _ = stdout.flush();
    }
}
