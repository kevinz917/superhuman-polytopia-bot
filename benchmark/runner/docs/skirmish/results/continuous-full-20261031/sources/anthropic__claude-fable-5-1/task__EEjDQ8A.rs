// Bardur 1v1 heuristic bot. One JSON observation per line; reply {"action_id":"..."}.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::cell::RefCell;
use std::collections::HashSet;
thread_local! {
    static BLOCKED: RefCell<HashSet<usize>> = RefCell::new(HashSet::new());
    static TECHN: RefCell<usize> = RefCell::new(0);
}

#[derive(Clone, Copy, Default)]
struct Tile { known: bool, terrain: i32, owner: i32, resource: i32, structure: i32, capital: i32 }
#[derive(Clone)]
struct Unit { tile: usize, owner: i32, kind: i32, hp: f64, moved: bool, attacked: bool, veteran: bool, home: i64 }
#[derive(Clone)]
struct City { tile: usize, owner: i32, level: i32 }

#[derive(Clone)]
struct Act { kind: i64, source: i64, target: i64, param: i64, id: String }

struct G {
    size: usize, n: usize, me: i32, turn: i64, stars: i64,
    tiles: Vec<Tile>, units: Vec<Unit>, cities: Vec<City>, tech: Vec<i64>, acts: Vec<Act>,
    unit_at: Vec<i32>, // index into units or -1
}

// cost, hp, atk, def, range, dash, escape
fn stats(kind: i32) -> (f64, f64, f64, f64, i32, bool, bool) {
    match kind {
        2 => (2.0, 10.0, 2.0, 2.0, 1, true, false),
        3 => (3.0, 10.0, 2.0, 1.0, 1, true, true),
        4 => (8.0, 10.0, 3.5, 1.0, 1, true, false),
        5 => (3.0, 15.0, 1.0, 3.0, 1, false, false),
        8 => (8.0, 10.0, 4.0, 0.0, 3, false, false),
        9 => (3.0, 10.0, 2.0, 1.0, 2, true, false),
        10 => (5.0, 10.0, 0.0, 1.0, 1, false, false),
        11 => (5.0, 15.0, 3.0, 3.0, 1, true, false),
        12 => (10.0, 40.0, 5.0, 4.0, 1, false, false),
        37 => (2.0, 10.0, 2.0, 2.0, 1, true, false),
        38 => (8.0, 5.0, 0.0, 0.5, 1, true, false),
        39 => (2.0, 5.0, 0.0, 0.5, 1, true, false),
        40 => (2.0, 10.0, 2.0, 2.0, 1, true, false),
        41 => (15.0, 10.0, 3.0, 2.0, 3, false, false),
        42 => (5.0, 10.0, 2.0, 1.0, 2, true, false),
        43 => (0.0, 10.0, 0.0, 1.0, 2, false, false),
        44 => (5.0, 10.0, 3.0, 3.0, 1, true, false),
        45 => (10.0, 40.0, 4.0, 4.0, 1, false, false),
        _ => (5.0, 10.0, 2.0, 1.0, 1, true, false),
    }
}

fn combat(aa: f64, ah: f64, am: f64, dd: f64, dh: f64, dm: f64, bonus: f64) -> (f64, f64) {
    let af = aa * ah / am.max(1.0);
    let df = dd * dh / dm.max(1.0) * bonus;
    let tot = af + df;
    if tot <= 0.0 { return (0.0, 0.0); }
    let a = ((af / tot) * aa * 4.5 + 0.5).floor();
    let r = if dh > a { ((df / tot) * dd * 4.5 + 0.5).floor() } else { 0.0 };
    (a, r)
}

impl G {
    fn parse(obs: &Value) -> G {
        let vs = &obs["visible_state"];
        let size = vs["board_size"].as_u64().unwrap_or(14) as usize;
        let n = size * size;
        let me = obs["player"].as_i64().unwrap_or(1) as i32;
        let turn = vs["turn"].as_i64().unwrap_or(0);
        let stars = vs["stars"].as_i64().unwrap_or(0);
        let mut tiles = vec![Tile::default(); n];
        if let Some(arr) = vs["tiles"].as_array() {
            for t in arr {
                let i = t["tile"].as_u64().unwrap_or(0) as usize;
                if i >= n { continue; }
                let known = t["known"].as_bool().unwrap_or(false);
                tiles[i] = Tile {
                    known,
                    terrain: t["terrain"].as_i64().unwrap_or(0) as i32,
                    owner: t["owner"].as_i64().unwrap_or(0) as i32,
                    resource: t["resource"].as_i64().unwrap_or(0) as i32,
                    structure: t["structure"].as_i64().unwrap_or(0) as i32,
                    capital: t["capital"].as_i64().unwrap_or(0) as i32,
                };
            }
        }
        let mut units = Vec::new();
        let mut unit_at = vec![-1i32; n];
        if let Some(arr) = vs["units"].as_array() {
            for u in arr {
                let tile = u["tile"].as_u64().unwrap_or(0) as usize;
                if tile >= n { continue; }
                unit_at[tile] = units.len() as i32;
                units.push(Unit {
                    tile,
                    owner: u["owner"].as_i64().unwrap_or(0) as i32,
                    kind: u["kind"].as_i64().unwrap_or(2) as i32,
                    hp: u["health"].as_f64().unwrap_or(10.0),
                    moved: u["moved"].as_bool().unwrap_or(false),
                    attacked: u["attacked"].as_bool().unwrap_or(false),
                    veteran: u["veteran"].as_bool().unwrap_or(false),
                    home: u["home"].as_i64().unwrap_or(0),
                });
            }
        }
        let mut cities = Vec::new();
        if let Some(arr) = vs["cities"].as_array() {
            for c in arr {
                let tile = c["tile"].as_u64().unwrap_or(0) as usize;
                if tile >= n { continue; }
                cities.push(City { tile, owner: c["owner"].as_i64().unwrap_or(0) as i32, level: c["level"].as_i64().unwrap_or(1) as i32 });
            }
        }
        let tech: Vec<i64> = vs["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
        let mut acts = Vec::new();
        if let Some(arr) = obs["legal_actions"].as_array() {
            for a in arr {
                let c = &a["command"];
                acts.push(Act {
                    kind: c["kind"].as_i64().unwrap_or(0),
                    source: c["source"].as_i64().unwrap_or(-1),
                    target: c["target"].as_i64().unwrap_or(-1),
                    param: c["parameter"].as_i64().unwrap_or(0),
                    id: a["action_id"].as_str().unwrap_or("").to_string(),
                });
            }
        }
        G { size, n, me, turn, stars, tiles, units, cities, tech, acts, unit_at }
    }

    fn d(&self, a: usize, b: usize) -> i32 {
        let s = self.size as i32;
        let (ax, ay) = ((a as i32) % s, (a as i32) / s);
        let (bx, by) = ((b as i32) % s, (b as i32) / s);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn has(&self, t: i64) -> bool { self.tech.contains(&t) }
    fn enemy_unit_at(&self, t: usize) -> Option<&Unit> {
        let i = self.unit_at[t];
        if i >= 0 { let u = &self.units[i as usize]; if u.owner != self.me && u.owner > 0 { return Some(u); } }
        None
    }
    fn own_unit_at(&self, t: usize) -> bool {
        let i = self.unit_at[t];
        i >= 0 && self.units[i as usize].owner == self.me
    }
    fn adj_enemies(&self, t: usize) -> i32 {
        let mut c = 0;
        for u in &self.units { if u.owner != self.me && u.owner > 0 && self.d(u.tile, t) == 1 { c += 1; } }
        c
    }
    fn near_enemy(&self, t: usize, r: i32) -> bool {
        self.units.iter().any(|u| u.owner != self.me && u.owner > 0 && self.d(u.tile, t) <= r)
    }
    fn defense_bonus(&self, u: &Unit) -> f64 {
        let t = &self.tiles[u.tile];
        let (_, _, _, _, _, _, _) = stats(u.kind);
        if t.structure == 1 && t.owner == u.owner { 1.5 }
        else if t.terrain == 5 || t.terrain == 4 || t.terrain == 1 || t.terrain == 2 { 1.3 }
        else { 1.0 }
    }
    fn capturable(&self, t: usize) -> bool {
        let tl = &self.tiles[t];
        tl.known && ((tl.structure == 1 && tl.owner != self.me) || tl.structure == 2)
    }
}

fn research_order(g: &G) -> Vec<i64> {
    // count water near own cities
    let mut water = 0;
    for c in g.cities.iter().filter(|c| c.owner == g.me) {
        for t in 0..g.n { if g.tiles[t].known && (g.tiles[t].terrain == 1 || g.tiles[t].terrain == 2) && g.d(c.tile, t) <= 2 { water += 1; } }
    }
    let mut v: Vec<i64> = vec![1, 6, 16, 7, 18, 20, 23, 24, 8, 2, 3, 17, 9, 13, 10, 22, 4, 5, 19, 12];
    if water >= 3 {
        v = vec![1, 6, 10, 16, 7, 18, 13, 20, 23, 24, 8, 2, 3, 17, 9, 22, 4, 5, 19, 12];
    }
    v
}

fn choose(obs: &Value) -> String {
    let g = G::parse(obs);
    if g.acts.is_empty() { return String::new(); }
    let fallback = g.acts.iter().find(|a| a.kind == 10).map(|a| a.id.clone()).unwrap_or_else(|| g.acts[0].id.clone());

    // 1. city rewards
    let rewards: Vec<&Act> = g.acts.iter().filter(|a| a.kind == 9).collect();
    if !rewards.is_empty() {
        let pref: Vec<i64> = if g.turn <= 10 { vec![4, 3, 7, 1, 8, 5, 6, 2] } else { vec![3, 4, 7, 1, 8, 5, 6, 2] };
        for p in pref { if let Some(a) = rewards.iter().find(|a| a.param == p) { return a.id.clone(); } }
        return rewards[0].id.clone();
    }
    // 2. capture
    if let Some(a) = g.acts.iter().find(|a| a.kind == 8) { return a.id.clone(); }
    // 3. promote
    if let Some(a) = g.acts.iter().find(|a| a.kind == 3 && a.param == 16) { return a.id.clone(); }

    // 4. attacks
    let mut best_att: Option<(f64, String)> = None;
    for a in g.acts.iter().filter(|a| a.kind == 2) {
        let (s, t) = (a.source as usize, a.target as usize);
        if s >= g.n || t >= g.n { continue; }
        let (si, ti) = (g.unit_at[s], g.unit_at[t]);
        if si < 0 || ti < 0 { continue; }
        let (au, du) = (&g.units[si as usize], &g.units[ti as usize]);
        if au.kind == 10 || au.kind == 38 { continue; }
        let (_, amax, aa, _, arange, _, _) = stats(au.kind);
        let (dcost, dmax, datk, dd, drange, _, _) = stats(du.kind);
        let amax = amax + if au.veteran { 5.0 } else { 0.0 };
        let dmax = dmax + if du.veteran { 5.0 } else { 0.0 };
        let bonus = g.defense_bonus(du);
        let (dmg, ret) = combat(aa, au.hp, amax, dd, du.hp, dmax, bonus);
        let kill = du.hp - dmg <= 0.0;
        let dist = g.d(s, t);
        let ret = if kill || dist > drange || datk <= 0.0 { 0.0 } else { ret };
        let die = au.hp - ret <= 0.0;
        let mut score = dmg + if kill { 8.0 + dcost * 1.5 } else { 0.0 } - ret * 0.7 - if die { 25.0 } else { 0.0 };
        if g.tiles[t].structure == 1 && g.tiles[t].owner == g.me { score += 4.0; } // defend our city
        if arange >= 2 { score += 1.0; }
        if !kill && ret >= dmg { score -= 3.0; }
        if score > 0.5 && (best_att.is_none() || score > best_att.as_ref().unwrap().0) { best_att = Some((score, a.id.clone())); }
    }
    if let Some((_, id)) = best_att { return id; }

    // army status
    let my_units = g.units.iter().filter(|u| u.owner == g.me).count() as i64;
    let my_cities = g.cities.iter().filter(|c| c.owner == g.me).count() as i64;
    let threatened = g.cities.iter().any(|c| c.owner == g.me && {
        let en = g.units.iter().filter(|u| u.owner != g.me && u.owner > 0 && g.d(u.tile, c.tile) <= 3).count();
        let own = g.units.iter().filter(|u| u.owner == g.me && g.d(u.tile, c.tile) <= 2).count();
        en > 0 && en >= own
    });
    let enemy_seen = g.units.iter().filter(|u| u.owner != g.me && u.owner > 0).count() as i64;
    let unclaimed = (0..g.n).filter(|&t| g.capturable(t)).count() as i64;
    let unknown = (0..g.n).filter(|&t| !g.tiles[t].known).count() as i64;
    let expanding = g.turn < 30 && (unclaimed > 0 || unknown > g.n as i64 / 6);

    let summon = |g: &G| -> Option<String> {
        let sums: Vec<&Act> = g.acts.iter().filter(|a| a.kind == 4 && a.param != 43 && a.param != 41 && a.param != 42 && a.param != 44 && a.param != 10 && a.param != 38).collect();
        if sums.is_empty() { return None; }
        let riders = g.units.iter().filter(|u| u.owner == g.me && u.kind == 3).count() as i64;
        let archers = g.units.iter().filter(|u| u.owner == g.me && u.kind == 9).count() as i64;
        let mut pref: Vec<i64> = Vec::new();
        let war = threatened || enemy_seen > 0;
        if war {
            if g.stars >= 8 && my_units >= 2 { pref.push(4); }
            if g.stars >= 5 { pref.push(11); }
            if threatened { pref.push(5); }
            if archers * 2 < my_units { pref.push(9); }
            pref.push(2); pref.push(5);
            if expanding && riders < my_cities && !threatened { pref.push(3); }
        } else {
            if expanding && riders < my_cities + 2 { pref.push(3); }
            if g.stars >= 5 && my_units >= my_cities + 2 { pref.push(11); }
            if archers * 3 < my_units { pref.push(9); }
        }
        pref.push(2); pref.push(3); pref.push(9); pref.push(5);
        for p in pref {
            let cands: Vec<&&Act> = sums.iter().filter(|a| a.param == p).collect();
            if cands.is_empty() { continue; }
            let mut best: Option<(i32, String)> = None;
            for a in cands {
                let t = a.source as usize;
                if t >= g.n { continue; }
                let mut score = 0;
                if g.near_enemy(t, 4) { score += 10; }
                let mut mind = 99;
                if p == 3 {
                    for i in 0..g.n { if g.capturable(i) || !g.tiles[i].known { mind = mind.min(g.d(t, i)); } }
                } else {
                    for c in &g.cities { if c.owner != g.me && c.owner > 0 { mind = mind.min(g.d(t, c.tile)); } }
                    for u in &g.units { if u.owner != g.me && u.owner > 0 { mind = mind.min(g.d(t, u.tile)); } }
                }
                score -= mind.min(20);
                if best.is_none() || score > best.as_ref().unwrap().0 { best = Some((score, a.id.clone())); }
            }
            if let Some((_, id)) = best { return Some(id); }
        }
        None
    };

    let res: Vec<&Act> = g.acts.iter().filter(|a| a.kind == 7).collect();
    // Riding first
    if let Some(a) = res.iter().find(|a| a.param == 1) { return a.id.clone(); }

    let army_ok = my_units >= 2 * my_cities && !threatened && my_units >= enemy_seen;
    let order = research_order(&g);
    if army_ok && g.tech.len() < 10 {
        for t in &order { if let Some(a) = res.iter().find(|a| a.param == *t) { return a.id.clone(); } }
    }
    if !threatened && my_units >= my_cities + 1 {
    // 5. harvest
    if let Some(a) = g.acts.iter().find(|a| a.kind == 5) { return a.id.clone(); }

    // 6. builds
    let mut best_build: Option<(f64, String)> = None;
    for a in g.acts.iter().filter(|a| a.kind == 6) {
        let t = a.target as usize;
        if t >= g.n { continue; }
        let adj_count = |kind: i32| -> i32 {
            let mut c = 0;
            for j in 0..g.n { if g.d(t, j) == 1 && g.tiles[j].structure == kind && g.tiles[j].owner == g.me { c += 1; } }
            c
        };
        let (pop, cost): (f64, f64) = match a.param {
            12 => (1.0, 3.0),
            5 => (2.0, 5.0),
            21 => (2.0, 5.0),
            13 => (adj_count(12) as f64, 5.0),
            6 => (adj_count(5) as f64, 5.0),
            22 => (2.0 * adj_count(21) as f64, 5.0),
            8 => { if g.turn >= 4 && g.stars >= 9 { (1.2, 7.0) } else { continue; } }
            50 => { let v = (adj_count(13) + adj_count(6) + adj_count(22)) as f64; if v >= 2.0 { (v * 1.5, 5.0) } else { continue; } }
            17 | 18 | 19 | 20 => { if g.stars >= 40 { (1.0, 20.0) } else { continue; } }
            23 | 24 | 25 | 26 | 27 | 28 | 29 => (3.0, 0.5),
            _ => continue,
        };
        let ratio = pop / cost;
        if ratio < 0.16 { continue; }
        if best_build.is_none() || ratio > best_build.as_ref().unwrap().0 { best_build = Some((ratio, a.id.clone())); }
    }
    if let Some((_, id)) = best_build { return id; }

    }
    if let Some(id) = summon(&g) { return id; }
    // 5. harvest
    if let Some(a) = g.acts.iter().find(|a| a.kind == 5) { return a.id.clone(); }

    // 6. builds
    let mut best_build: Option<(f64, String)> = None;
    for a in g.acts.iter().filter(|a| a.kind == 6) {
        let t = a.target as usize;
        if t >= g.n { continue; }
        let adj_count = |kind: i32| -> i32 {
            let mut c = 0;
            for j in 0..g.n { if g.d(t, j) == 1 && g.tiles[j].structure == kind && g.tiles[j].owner == g.me { c += 1; } }
            c
        };
        let (pop, cost): (f64, f64) = match a.param {
            12 => (1.0, 3.0),
            5 => (2.0, 5.0),
            21 => (2.0, 5.0),
            13 => (adj_count(12) as f64, 5.0),
            6 => (adj_count(5) as f64, 5.0),
            22 => (2.0 * adj_count(21) as f64, 5.0),
            8 => { if g.turn >= 4 && g.stars >= 9 { (1.2, 7.0) } else { continue; } }
            50 => { let v = (adj_count(13) + adj_count(6) + adj_count(22)) as f64; if v >= 2.0 { (v * 1.5, 5.0) } else { continue; } }
            17 | 18 | 19 | 20 => { if g.stars >= 40 { (1.0, 20.0) } else { continue; } }
            23 | 24 | 25 | 26 | 27 | 28 | 29 => (3.0, 0.5),
            _ => continue,
        };
        let ratio = pop / cost;
        if ratio < 0.16 { continue; }
        if best_build.is_none() || ratio > best_build.as_ref().unwrap().0 { best_build = Some((ratio, a.id.clone())); }
    }
    if let Some((_, id)) = best_build { return id; }

    for t in &order { if let Some(a) = res.iter().find(|a| a.param == *t) { return a.id.clone(); } }

    // 9. moves
    // goals
    let enemy_capital: Option<usize> = (0..g.n).find(|&i| g.tiles[i].known && g.tiles[i].capital > 0 && g.tiles[i].capital != g.me);
    TECHN.with(|c| { let mut c = c.borrow_mut(); if *c != g.tech.len() { *c = g.tech.len(); BLOCKED.with(|b| b.borrow_mut().clear()); } });
    let blocked: HashSet<usize> = BLOCKED.with(|b| b.borrow().clone());
    let can_mountain = g.tech.contains(&20);
    let can_water = g.tech.contains(&10);
    let mut goals: Vec<(usize, f64)> = Vec::new();
    for t in 0..g.n {
        let tl = &g.tiles[t];
        if !tl.known { continue; }
        if blocked.contains(&t) { continue; }
        if tl.terrain == 4 && !can_mountain { continue; }
        if (tl.terrain == 1 || tl.terrain == 2) && !can_water { continue; }
        if g.capturable(t) && !g.own_unit_at(t) {
            let w = if tl.owner > 0 { 1.0 } else { 0.8 };
            goals.push((t, w));
        }
    }
    for c in g.cities.iter().filter(|c| c.owner == g.me) {
        if g.near_enemy(c.tile, 4) && !g.own_unit_at(c.tile) { goals.push((c.tile, 0.3)); }
    }
    for u in g.units.iter().filter(|u| u.owner != g.me && u.owner > 0) {
        if my_units >= enemy_seen { goals.push((u.tile, 1.6)); }
    }
    // frontier unknown tiles
    for t in 0..g.n {
        if g.tiles[t].known { continue; }
        let s = g.size as i32;
        let (x, y) = ((t as i32) % s, (t as i32) / s);
        let mut frontier = false;
        for dy in -1..=1 { for dx in -1..=1 {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && nx < s && ny >= 0 && ny < s && g.tiles[(ny * s + nx) as usize].known { frontier = true; }
        }}
        if frontier { goals.push((t, 2.2)); }
    }
    if let Some(ec) = enemy_capital { goals.push((ec, 1.4)); }

    let movers: Vec<usize> = (0..g.units.len()).filter(|&i| g.units[i].owner == g.me && !g.units[i].moved && g.acts.iter().any(|a| a.kind == 1 && a.source as usize == g.units[i].tile)).collect();
    // greedy assignment
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
    for &ui in &movers {
        let ut = g.units[ui].tile;
        for (gi, &(gt, w)) in goals.iter().enumerate() {
            pairs.push((g.d(ut, gt) as f64 * w, ui, gi));
        }
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut assigned_u = vec![usize::MAX; g.units.len()];
    let mut used_g = vec![false; goals.len()];
    for &(_, ui, gi) in &pairs {
        if assigned_u[ui] == usize::MAX && !used_g[gi] { assigned_u[ui] = gi; used_g[gi] = true; }
    }
    let ports: Vec<usize> = (0..g.n).filter(|&t| g.tiles[t].known && g.tiles[t].structure == 8 && g.tiles[t].owner == g.me).collect();
    let unknown_adj = |t: usize| -> i32 {
        let s = g.size as i32;
        let (x, y) = ((t as i32) % s, (t as i32) / s);
        let mut c = 0;
        for dy in -1..=1 { for dx in -1..=1 {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && nx < s && ny >= 0 && ny < s && !g.tiles[(ny * s + nx) as usize].known { c += 1; }
        }}
        c
    };
    let is_boat = |k: i32| k == 43 || k == 44 || k == 42 || k == 41;
    for &ui in &movers {
        let u = &g.units[ui];
        let ut = u.tile;
        let on_own_city = g.tiles[ut].structure == 1 && g.tiles[ut].owner == g.me;
        if on_own_city && g.near_enemy(ut, 3) {
            // step aside only to let the city train a replacement garrison
            let level = g.cities.iter().find(|c| c.tile == ut).map(|c| c.level).unwrap_or(1) as i64;
            let homed = g.units.iter().filter(|x| x.owner == g.me && x.home == ut as i64 + 1).count() as i64;
            if g.stars >= 2 && homed <= level {
                let mut best: Option<(f64, String)> = None;
                for a in g.acts.iter().filter(|a| a.kind == 1 && a.source as usize == ut && (a.target as usize) < g.n) {
                    let t = a.target as usize;
                    if g.d(t, ut) != 1 || g.adj_enemies(t) > 0 { continue; }
                    let tt = &g.tiles[t];
                    let mut sc = -(g.units.iter().filter(|e| e.owner != g.me && e.owner > 0 && g.d(e.tile, t) <= 2).count() as f64) * 3.0;
                    if tt.terrain == 5 || tt.terrain == 4 { sc += 2.0; }
                    if best.is_none() || sc > best.as_ref().unwrap().0 { best = Some((sc, a.id.clone())); }
                }
                if let Some((_, id)) = best { return id; }
            }
            continue;
        }
        let (_, umax, _, _, _, _, _) = stats(u.kind);
        let hpfrac = u.hp / (umax + if u.veteran { 5.0 } else { 0.0 });
        let moves_all: Vec<(usize, &Act)> = g.acts.iter().filter(|a| a.kind == 1 && a.source as usize == ut && (a.target as usize) < g.n).map(|a| (a.target as usize, a)).collect();
        if moves_all.is_empty() { continue; }
        // retreat: after attacking (escape units) or when wounded near enemies
        let near3 = g.near_enemy(ut, 3);
        if (u.attacked || hpfrac < 0.5) && near3 {
            let own_city_d = |t: usize| g.cities.iter().filter(|c| c.owner == g.me).map(|c| g.d(t, c.tile)).min().unwrap_or(10);
            let safety = |t: usize| -> f64 {
                let th = g.units.iter().filter(|e| e.owner != g.me && e.owner > 0 && g.d(e.tile, t) <= 2).count() as f64;
                let mut sc = -th * 4.0 - (g.adj_enemies(t) as f64) * 6.0 - own_city_d(t) as f64 * 1.5;
                let tt = &g.tiles[t];
                if tt.owner == g.me { sc += 3.0; }
                if tt.terrain == 5 || tt.terrain == 4 { sc += 1.5; }
                if tt.structure == 1 && tt.owner == g.me { sc += 3.0; }
                if g.capturable(t) { sc += 10.0; }
                sc
            };
            let cur = safety(ut);
            let mut best: Option<(f64, String)> = None;
            for (t, a) in &moves_all {
                let sc = safety(*t);
                if sc > cur + 1.0 && (best.is_none() || sc > best.as_ref().unwrap().0) { best = Some((sc, a.id.clone())); }
            }
            if let Some((_, id)) = best { return id; }
            continue;
        }
        // goal list: assigned first, then nearest unused goals
        let mut glist: Vec<usize> = Vec::new();
        if assigned_u[ui] != usize::MAX { glist.push(goals[assigned_u[ui]].0); }
        let mut others: Vec<(f64, usize)> = goals.iter().enumerate().filter(|(gi, _)| !used_g[*gi] || Some(*gi) == Some(assigned_u[ui])).map(|(_, &(gt, w))| (g.d(ut, gt) as f64 * w, gt)).collect();
        others.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, gt) in others.into_iter().take(10) { if !glist.contains(&gt) { glist.push(gt); } }
        if glist.is_empty() { if let Some(ec) = enemy_capital { glist.push(ec); } }
        let local = |target: usize| -> f64 {
            let mut sc = 0.0;
            if g.capturable(target) { sc += 30.0; }
            let tt = &g.tiles[target];
            if tt.terrain == 5 || tt.terrain == 4 { sc += 1.0; }
            if tt.structure == 1 && tt.owner == g.me { if g.near_enemy(target, 4) { sc += 6.0; } else { sc -= 1.0; } }
            sc += unknown_adj(target) as f64 * 1.5;
            let ae = g.adj_enemies(target) as f64;
            if hpfrac < 0.5 { sc -= ae * 8.0; } else if ae > 0.0 { sc += 3.0 - (ae - 1.0) * 5.0; }
            let th = g.units.iter().filter(|e| e.owner != g.me && e.owner > 0 && g.d(e.tile, target) <= 2).count() as f64;
            if th >= 2.0 { sc -= (th - 1.0) * 3.0; }
            if hpfrac < 0.5 { sc -= th * 3.0; }
            sc
        };
        let moves: Vec<(usize, &Act)> = g.acts.iter().filter(|a| a.kind == 1 && a.source as usize == ut && (a.target as usize) < g.n).map(|a| (a.target as usize, a)).collect();
        if moves.is_empty() { continue; }
        // capturable directly reachable
        if let Some((_, a)) = moves.iter().filter(|(t, _)| g.capturable(*t)).min_by_key(|(t, _)| g.adj_enemies(*t)) { return a.id.clone(); }
        let mut chosen: Option<String> = None;
        for gt in &glist {
            let cur_d = g.d(ut, *gt);
            if cur_d == 1 && g.unit_at[*gt] < 0 && !moves.iter().any(|(t, _)| t == gt) {
                BLOCKED.with(|b| b.borrow_mut().insert(*gt));
                continue;
            }
            let mut best: Option<(f64, String)> = None;
            for (t, a) in &moves {
                let d = g.d(*t, *gt);
                if d >= cur_d { continue; }
                let sc = -(d as f64) * 100.0 + local(*t);
                if best.is_none() || sc > best.as_ref().unwrap().0 { best = Some((sc, a.id.clone())); }
            }
            if let Some((_, id)) = best { chosen = Some(id); break; }
        }
        if let Some(id) = chosen { return id; }
        // port fallback for land units blocked by water
        if !is_boat(u.kind) && !ports.is_empty() && !glist.is_empty() {
            let cd = ports.iter().map(|&p| g.d(ut, p)).min().unwrap_or(99);
            let mut pb: Option<(i32, String)> = None;
            for (t, a) in &moves {
                let pd = ports.iter().map(|&p| g.d(*t, p)).min().unwrap_or(99);
                if pd < cd && (pb.is_none() || pd < pb.as_ref().unwrap().0) { pb = Some((pd, a.id.clone())); }
            }
            if let Some((_, id)) = pb { return id; }
        }
        // explore: move revealing most unknown tiles
        let mut eb: Option<(f64, String)> = None;
        for (t, a) in &moves {
            let sc = local(*t);
            if unknown_adj(*t) > 0 && (eb.is_none() || sc > eb.as_ref().unwrap().0) { eb = Some((sc, a.id.clone())); }
        }
        if let Some((_, id)) = eb { return id; }
    }

    fallback
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => Value::Null };
        let mut id = choose(&obs);
        if id.is_empty() {
            if let Some(a) = obs["legal_actions"].as_array().and_then(|a| a.first()) { id = a["action_id"].as_str().unwrap_or("").to_string(); }
        }
        let _ = writeln!(stdout, "{}", json!({"action_id": id}));
        let _ = stdout.flush();
    }
}
