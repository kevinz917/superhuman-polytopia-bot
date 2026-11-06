// Polytopia Bardur 1v1 bot: greedy priority-chain strategy with BFS pathing.
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

#[derive(Clone)]
struct Legal {
    id: String,
    kind: i32,
    source: i32,
    target: i32,
    parameter: i32,
}

struct U {
    tile: usize,
    owner: i32,
    kind: i32,
    hp: f32,
    vet: bool,
    passenger: i32,
}

struct GS {
    size: usize,
    n: usize,
    me: i32,
    turn: i32,
    stars: i32,
    tech: [bool; 128],
    known: Vec<bool>,
    terr: Vec<i32>,
    owner: Vec<i32>,
    struc: Vec<i32>,
    res: Vec<i32>,
    units: Vec<U>,
    unit_at: Vec<i32>,
    cities: Vec<(usize, i32, Vec<i32>)>, // tile, owner, rewards
    city_at: Vec<i32>,
    legal: Vec<Legal>,
    nbrs: Vec<Vec<usize>>,
}

// (cost, attack, defense, hp, range)
fn stats(kind: i32) -> (i32, f32, f32, f32, i32) {
    match kind {
        2 => (2, 2.0, 2.0, 10.0, 1),
        3 => (3, 2.0, 1.0, 10.0, 1),
        4 => (8, 3.5, 1.0, 10.0, 1),
        5 => (3, 1.0, 3.0, 15.0, 1),
        8 => (8, 4.0, 0.0, 10.0, 3),
        9 => (3, 2.0, 1.0, 10.0, 2),
        10 => (5, 0.0, 1.0, 10.0, 1),
        11 => (5, 3.0, 3.0, 15.0, 1),
        12 => (10, 5.0, 4.0, 40.0, 1),
        15 => (2, 3.0, 1.0, 15.0, 1),
        37 => (2, 2.0, 2.0, 10.0, 1),
        38 => (8, 0.0, 0.5, 5.0, 1),
        39 => (2, 0.0, 0.5, 5.0, 1),
        40 => (2, 2.0, 2.0, 10.0, 1),
        41 => (15, 3.0, 2.0, 10.0, 3),
        42 => (5, 2.0, 1.0, 10.0, 2),
        43 => (0, 0.0, 1.0, 10.0, 2),
        44 => (5, 3.0, 3.0, 10.0, 1),
        45 => (10, 4.0, 4.0, 40.0, 1),
        _ => (5, 2.0, 2.0, 10.0, 1),
    }
}

fn stiff(kind: i32) -> bool {
    matches!(kind, 8 | 10 | 22 | 24 | 25 | 29 | 34 | 38 | 39 | 41 | 43 | 56 | 57 | 60 | 61 | 62)
}

fn is_boat(u: &U) -> bool {
    u.passenger > 0 || matches!(u.kind, 39 | 40 | 41 | 42 | 43 | 44 | 45)
}

impl GS {
    fn cheb(&self, a: usize, b: usize) -> i32 {
        let s = self.size as i32;
        let (ax, ay) = ((a % self.size) as i32, (a / self.size) as i32);
        let (bx, by) = ((b % self.size) as i32, (b / self.size) as i32);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn land_pass(&self, t: usize) -> bool {
        if !self.known[t] {
            return false;
        }
        let te = self.terr[t];
        if te >= 3 {
            return te != 4 || self.tech[20];
        }
        if self.struc[t] == 48 {
            return true;
        }
        self.struc[t] == 8 && self.owner[t] == self.me
    }
    fn boat_pass(&self, t: usize) -> bool {
        if !self.known[t] {
            return false;
        }
        let te = self.terr[t];
        if te == 4 {
            return self.tech[20];
        }
        if te == 1 {
            return self.tech[10];
        }
        if te == 2 {
            return self.tech[13];
        }
        true
    }
    fn bfs(&self, starts: &[usize], boat: bool) -> Vec<i32> {
        let n = self.n;
        let mut d = vec![-1i32; n];
        let mut q: VecDeque<usize> = VecDeque::new();
        let pass = |t: usize| {
            if boat {
                self.boat_pass(t)
            } else {
                self.land_pass(t)
            }
        };
        for &s in starts {
            if s < n && d[s] < 0 && pass(s) {
                d[s] = 0;
                q.push_back(s);
            }
        }
        while let Some(x) = q.pop_front() {
            let base = d[x] + 1;
            for &y in &self.nbrs[x] {
                if d[y] < 0 && pass(y) {
                    d[y] = base;
                    q.push_back(y);
                }
            }
        }
        d
    }
    fn own_cities(&self) -> Vec<usize> {
        self.cities
            .iter()
            .filter(|c| c.1 == self.me)
            .map(|c| c.0)
            .collect()
    }
}

fn ig(v: &Value, k: &str) -> i32 {
    v[k].as_i64().unwrap_or(0) as i32
}

fn parse(obs: &Value) -> GS {
    let vs = &obs["visible_state"];
    let size = vs["board_size"].as_i64().unwrap_or(14) as usize;
    let n = size * size;
    let me = obs["player"].as_i64().unwrap_or(1) as i32;
    let mut tech = [false; 128];
    if let Some(arr) = vs["technology"].as_array() {
        for t in arr {
            if let Some(k) = t.as_i64() {
                if k >= 0 && k < 128 {
                    tech[k as usize] = true;
                }
            }
        }
    }
    let mut known = vec![false; n];
    let mut terr = vec![3i32; n];
    let mut owner = vec![0i32; n];
    let mut struc = vec![0i32; n];
    let mut res = vec![0i32; n];
    if let Some(arr) = vs["tiles"].as_array() {
        for t in arr {
            let tile = t["tile"].as_i64().unwrap_or(-1) as usize;
            if tile >= n || t["known"].as_bool() != Some(true) {
                continue;
            }
            known[tile] = true;
            terr[tile] = ig(t, "terrain");
            owner[tile] = ig(t, "owner");
            struc[tile] = ig(t, "structure");
            res[tile] = ig(t, "resource");
        }
    }
    let mut units = Vec::new();
    let mut unit_at = vec![-1i32; n];
    if let Some(arr) = vs["units"].as_array() {
        for u in arr {
            let tile = u["tile"].as_i64().unwrap_or(-1) as usize;
            if tile >= n {
                continue;
            }
            unit_at[tile] = units.len() as i32;
            units.push(U {
                tile,
                owner: ig(u, "owner"),
                kind: ig(u, "kind"),
                hp: u["health"].as_f64().unwrap_or(10.0) as f32,
                vet: u["veteran"].as_bool().unwrap_or(false),
                passenger: ig(u, "passenger"),
            });
        }
    }
    let mut cities = Vec::new();
    let mut city_at = vec![-1i32; n];
    if let Some(arr) = vs["cities"].as_array() {
        for c in arr {
            let tile = c["tile"].as_i64().unwrap_or(-1) as usize;
            if tile >= n {
                continue;
            }
            city_at[tile] = cities.len() as i32;
            let mut rw = vec![0i32; 9];
            if let Some(r) = c["rewards"].as_array() {
                for (i, x) in r.iter().take(9).enumerate() {
                    rw[i] = x.as_i64().unwrap_or(0) as i32;
                }
            }
            cities.push((tile, ig(c, "owner"), rw));
        }
    }
    let mut legal = Vec::new();
    if let Some(arr) = obs["legal_actions"].as_array() {
        for a in arr {
            let c = &a["command"];
            legal.push(Legal {
                id: a["action_id"].as_str().unwrap_or("").to_string(),
                kind: ig(c, "kind"),
                source: ig(c, "source"),
                target: ig(c, "target"),
                parameter: ig(c, "parameter"),
            });
        }
    }
    let mut nbrs = vec![Vec::with_capacity(8); n];
    for t in 0..n {
        let (x, y) = ((t % size) as i32, (t / size) as i32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x + dx, y + dy);
                if nx >= 0 && nx < size as i32 && ny >= 0 && ny < size as i32 {
                    nbrs[t].push((ny * size as i32 + nx) as usize);
                }
            }
        }
    }
    GS {
        size,
        n,
        me,
        turn: vs["turn"].as_i64().unwrap_or(0) as i32,
        stars: vs["stars"].as_i64().unwrap_or(0) as i32,
        tech,
        known,
        terr,
        owner,
        struc,
        res,
        units,
        unit_at,
        cities,
        city_at,
        legal,
        nbrs,
    }
}

fn unknown_nbrs(gs: &GS, t: usize) -> i32 {
    gs.nbrs[t].iter().filter(|&&y| !gs.known[y]).count() as i32
}

fn choose(obs: &Value) -> String {
    let gs = parse(obs);
    if gs.legal.is_empty() {
        return "0".to_string();
    }
    let end = gs
        .legal
        .iter()
        .find(|l| l.kind == 10)
        .map(|l| l.id.clone())
        .unwrap_or_else(|| gs.legal[0].id.clone());

    // 1. mandatory city reward
    if let Some(id) = pick_reward(&gs) {
        return id;
    }
    // 2. free promotion
    if let Some(l) = gs.legal.iter().find(|l| l.kind == 3 && l.parameter == 16) {
        return l.id.clone();
    }
    // 3. capture villages / ruins
    if let Some(l) = gs.legal.iter().find(|l| l.kind == 8) {
        return l.id.clone();
    }
    // 4. attacks
    if let Some(id) = pick_attack(&gs) {
        return id;
    }

    // shared analysis
    let own_cities = gs.own_cities();
    let ncity = own_cities.len().max(1) as i32;
    let villages: Vec<usize> = (0..gs.n)
        .filter(|&t| gs.known[t] && gs.struc[t] == 1 && gs.owner[t] == 0)
        .collect();
    let ruins: Vec<usize> = (0..gs.n)
        .filter(|&t| gs.known[t] && gs.struc[t] == 2)
        .collect();
    let enemy_cities: Vec<usize> = (0..gs.n)
        .filter(|&t| gs.known[t] && gs.struc[t] == 1 && gs.owner[t] != 0 && gs.owner[t] != gs.me)
        .collect();
    let ports: Vec<usize> = (0..gs.n)
        .filter(|&t| gs.struc[t] == 8 && gs.owner[t] == gs.me)
        .collect();
    let goals: Vec<(usize, i32)> = villages
        .iter()
        .map(|&v| (v, 0))
        .chain(ruins.iter().map(|&r| (r, 3)))
        .collect();
    let known_count = gs.known.iter().filter(|&&k| k).count().max(1) as f32;
    let water_frac = (0..gs.n)
        .filter(|&t| gs.known[t] && gs.terr[t] <= 2)
        .count() as f32
        / known_count;

    // naval need: no land-reachable neutral goals from our cities, or enemy cut off
    let dland_city = gs.bfs(&own_cities, false);
    let dall_city = gs.bfs(&own_cities, true);
    let any_land_goal = goals.iter().any(|&(g, _)| dland_city[g] >= 0);
    let any_boat_goal = goals.iter().any(|&(g, _)| dall_city[g] >= 0);
    let enemy_known = !enemy_cities.is_empty();
    let land_to_enemy = enemy_known
        && enemy_cities
            .iter()
            .any(|&e| dland_city[e] >= 0 || gs.cheb(e, own_cities[0]) <= 3);
    let mut naval = water_frac >= 0.2 && !any_land_goal && any_boat_goal;
    if enemy_known && !land_to_enemy && water_frac >= 0.15 {
        naval = true;
    }
    if !enemy_known && gs.turn > 25 && water_frac >= 0.15 {
        naval = true;
    }

    // threat detection
    let threat = gs.units.iter().any(|u| {
        u.owner != gs.me
            && u.owner != 0
            && own_cities.iter().any(|&c| gs.cheb(u.tile, c) <= 5)
    });
    // 5. research
    if let Some(id) = pick_research(&gs, ncity, naval, threat) {
        return id;
    }
    let _wanted_cost = next_tech_cost(&gs, ncity, naval);
    // 6. harvest (population = city levels = unit capacity)
    if gs.stars >= 2 && (gs.stars >= 4 || gs.turn <= 1) {
        if let Some(l) = gs.legal.iter().find(|l| l.kind == 5) {
            return l.id.clone();
        }
    }
    // 7. build
    if let Some(id) = pick_build(&gs, naval, &ports, if goals.is_empty() { 4 } else { 1 }) {
        return id;
    }
    // 8. train
    if let Some(id) = pick_train(&gs, &goals, &enemy_cities, &dland_city, &dall_city, threat) {
        return id;
    }
    // 9. move
    if let Some(id) = pick_move(&gs, &goals, &enemy_cities, &ports, naval, threat) {
        return id;
    }
    end
}

fn pick_reward(gs: &GS) -> Option<String> {
    let mut best: Option<&Legal> = None;
    let mut best_rank = -1;
    // prefer wall when enemy near this city
    for l in gs.legal.iter().filter(|l| l.kind == 9) {
        let city_tile = l.target as usize;
        let enemy_near = gs.units.iter().any(|u| {
            u.owner != gs.me && u.owner != 0 && gs.cheb(u.tile, city_tile) <= 3
        });
        let mut rank = match l.parameter {
            3 => 9,
            6 => 8,
            8 => 7,
            7 => 6,
            5 => 5,
            1 => if enemy_near { 8 } else { 4 },
            2 => 3,
            4 => 2,
            _ => 0,
        };
        if l.parameter == 6 && gs.stars < 6 {
            rank = 3; // giant ok anyway; keep
        }
        if rank > best_rank {
            best_rank = rank;
            best = Some(l);
        }
    }
    best.map(|l| l.id.clone())
}

fn pick_attack(gs: &GS) -> Option<String> {
    let own_cities = gs.own_cities();
    let mut best_v = 1.0f32;
    let mut best_id: Option<String> = None;
    for l in gs.legal.iter().filter(|l| l.kind == 2) {
        let s = l.source as usize;
        let t = l.target as usize;
        if s >= gs.n || t >= gs.n {
            continue;
        }
        let ai = gs.unit_at[s];
        let di = gs.unit_at[t];
        if ai < 0 || di < 0 {
            continue;
        }
        let a = &gs.units[ai as usize];
        let d = &gs.units[di as usize];
        if a.owner != gs.me || d.owner == gs.me || d.owner == 0 {
            continue;
        }
        let akind = if a.passenger > 0 { a.passenger } else { a.kind };
        let (acost, aatk, _adef, ahp, _ar) = stats(akind);
        let (dcost, _datk, ddef, dhp, drange) = stats(d.kind);
        let amax = ahp + if a.vet { 5.0 } else { 0.0 };
        let dmax = dhp + if d.vet { 5.0 } else { 0.0 };
        let mut v;
        if a.kind == 10 {
            v = dcost as f32 * 2.0 + 6.0;
        } else {
            let mut bonus = 1.0f32;
            if gs.terr[t] == 4 || gs.terr[t] == 5 {
                bonus = 1.5;
            }
            if gs.struc[t] == 1 && gs.owner[t] == d.owner {
                let ci = gs.city_at[t];
                bonus = if ci >= 0 && gs.cities[ci as usize].2.get(1).copied().unwrap_or(0) > 0 {
                    4.0
                } else {
                    1.5
                };
            }
            let af = aatk * (a.hp / amax);
            let df = ddef * (d.hp / dmax) * bonus;
            let tot = af + df;
            let dmg = if tot > 0.0 {
                (af / tot * aatk * 4.5 + 0.5).floor()
            } else {
                0.0
            };
            let kills = dmg >= d.hp - 0.001;
            let dist = gs.cheb(s, t);
            let ret = if kills || stiff(d.kind) || drange < dist || _datk <= 0.0 {
                0.0
            } else {
                (df / tot * _datk * 4.5 + 0.5).floor()
            };
            v = dmg;
            if kills {
                v += 10.0 + dcost as f32 * 1.5;
            }
            v -= ret * 0.8;
            if a.hp - ret <= 0.0 {
                v -= 8.0 + acost as f32 * 1.2;
            }
        }
        if own_cities.iter().any(|&c| gs.cheb(c, t) <= 2) {
            v += 3.0;
        }
        if v > best_v {
            best_v = v;
            best_id = Some(l.id.clone());
        }
    }
    best_id
}

fn tech_tier(k: i32) -> i32 {
    match k {
        6 | 1 | 20 | 10 => 1,
        18 | 8 | 23 | 16 | 13 | 7 => 2,
        _ => 3,
    }
}

fn wanted_list(naval: bool) -> Vec<i32> {
    let mut v: Vec<i32> = Vec::new();
    if naval {
        v.push(10);
        v.push(13);
    }
    v.extend([6, 1, 16, 20, 18, 7, 23, 24, 8, 3, 17, 22]);
    v
}

fn next_tech_cost(gs: &GS, ncity: i32, naval: bool) -> i32 {
    for k in wanted_list(naval) {
        if !gs.tech[k as usize] {
            return 4 + ncity * tech_tier(k);
        }
    }
    0
}

fn pick_research(gs: &GS, ncity: i32, naval: bool, threat: bool) -> Option<String> {
    let our_units = gs.units.iter().filter(|u| u.owner == gs.me).count() as i32;
    let army_ok = our_units >= ncity + 3 || gs.turn > 12;
    for k in wanted_list(naval) {
        if gs.tech[k as usize] {
            continue;
        }
        if !army_ok && !(naval && k == 10) {
            return None;
        }
        let cost = 4 + ncity * tech_tier(k);
        let buffer = if threat { 2 } else { 4 };
        if gs.stars >= cost + buffer {
            if let Some(l) = gs.legal.iter().find(|l| l.kind == 7 && l.parameter == k) {
                return Some(l.id.clone());
            }
        }
        return None;
    }
    None
}

fn pick_build(gs: &GS, naval: bool, ports: &[usize], max_ports: usize) -> Option<String> {
    if naval && gs.tech[10] && ports.len() < max_ports && gs.stars >= 9 {
        if let Some(l) = gs.legal.iter().find(|l| l.kind == 6 && l.parameter == 8) {
            return Some(l.id.clone());
        }
    }
    let order: &[(i32, i32)] = &[
        (12, 5),  // lumber hut
        (5, 7),   // farm
        (21, 7),  // mine
        (13, 11), // sawmill
        (22, 11), // forge
        (6, 11),  // windmill
    ];
    for &(param, need) in order {
        if gs.stars >= need {
            if let Some(l) = gs.legal.iter().find(|l| l.kind == 6 && l.parameter == param) {
                return Some(l.id.clone());
            }
        }
    }
    None
}

fn pick_train(
    gs: &GS,
    goals: &[(usize, i32)],
    enemy_cities: &[usize],
    dland_city: &[i32],
    dall_city: &[i32],
    threat: bool,
) -> Option<String> {
    let our_units = gs.units.iter().filter(|u| u.owner == gs.me).count() as i32;
    let ncity = gs.own_cities().len() as i32;
    let enemy_close = gs
        .units
        .iter()
        .any(|u| u.owner != gs.me && u.owner != 0 && gs.cheb(u.tile, gs.own_cities()[0]) <= 6);
    let war = enemy_close
        || (!enemy_cities.is_empty() && goals.is_empty())
        || gs.turn > 40;
    let goals_reachable = goals
        .iter()
        .filter(|g| dland_city[g.0] >= 0 || dall_city[g.0] >= 0)
        .count() as i32;
    let desired = if threat {
        our_units + 5
    } else if war {
        12 + ncity
    } else {
        (goals_reachable + 3).max(3 + 2 * ncity)
    }
    .min(20);
    if our_units >= desired {
        return None;
    }
    let mut prefs: Vec<i32> = Vec::new();
    if war || threat {
        if gs.tech[24] {
            prefs.extend([11, 9, 3, 2]);
        } else if gs.tech[18] {
            prefs.extend([9, 3, 2]);
        } else if gs.tech[1] {
            prefs.extend([3, 2]);
        } else {
            prefs.push(2);
        }
    } else if gs.tech[1] {
        prefs.extend([3, 2]);
    } else {
        prefs.push(2);
    }
    let enemy_tiles: Vec<usize> = gs
        .units
        .iter()
        .filter(|u| u.owner != gs.me && u.owner != 0)
        .map(|u| u.tile)
        .collect();
    for p in prefs {
        let cost = stats(p).0;
        if gs.stars < cost + 1 {
            continue;
        }
        let cands: Vec<&Legal> = gs
            .legal
            .iter()
            .filter(|l| l.kind == 4 && l.parameter == p)
            .collect();
        if cands.is_empty() {
            continue;
        }
        // prefer city closest to a goal / enemy
        let mut starts: Vec<usize> = goals.iter().map(|g| g.0).collect();
        starts.extend_from_slice(enemy_cities);
        starts.extend_from_slice(&enemy_tiles);
        let starts = if starts.is_empty() { gs.own_cities() } else { starts };
        let d = gs.bfs(&starts, false);
        let best = cands
            .iter()
            .min_by_key(|l| d[l.source as usize].max(-1))
            .unwrap();
        return Some(best.id.clone());
    }
    None
}

fn pick_move(
    gs: &GS,
    goals: &[(usize, i32)],
    enemy_cities: &[usize],
    ports: &[usize],
    naval: bool,
    threat: bool,
) -> Option<String> {
    if goals.is_empty() && enemy_cities.is_empty() {
        // still explore
    }
    // distance maps per goal
    let mut goal_land: Vec<Vec<i32>> = Vec::new();
    let mut goal_all: Vec<Vec<i32>> = Vec::new();
    for &(g, _) in goals {
        goal_land.push(gs.bfs(&[g], false));
        goal_all.push(gs.bfs(&[g], true));
    }
    // frontier
    let mut front: Vec<usize> = Vec::new();
    for t in 0..gs.n {
        if (gs.land_pass(t) || gs.boat_pass(t)) && gs.nbrs[t].iter().any(|&y| !gs.known[y]) {
            front.push(t);
        }
    }
    let front_land = gs.bfs(&front, false);
    let front_all = gs.bfs(&front, true);
    let mut estart: Vec<usize> = enemy_cities.to_vec();
    estart.extend(
        gs.units
            .iter()
            .filter(|u| u.owner != gs.me && u.owner != 0)
            .map(|u| u.tile),
    );
    let enemy_land = gs.bfs(&estart, false);
    let enemy_all = gs.bfs(&estart, true);
    let port_land = gs.bfs(ports, false);

    // movable units: sources of legal move actions
    let mut movable: Vec<usize> = Vec::new();
    for l in gs.legal.iter().filter(|l| l.kind == 1) {
        let s = l.source as usize;
        if s < gs.n && !movable.contains(&s) {
            movable.push(s);
        }
    }
    // role assignment
    let mut claimed = vec![false; goals.len()];
    #[derive(Clone, Copy)]
    enum Role {
        Goal(usize),
        Port,
        Front,
        Enemy,
        Idle,
    }
    let mut roles = vec![Role::Idle; gs.n];
    // order units by distance to nearest goal for better claims
    let mut order: Vec<usize> = movable.clone();
    order.sort_by_key(|&s| {
        let ui = gs.unit_at[s];
        if ui < 0 {
            return 9999;
        }
        let u = &gs.units[ui as usize];
        let maps = if is_boat(u) { &goal_all } else { &goal_land };
        let mut best = 9999;
        for m in maps {
            let v = m[s];
            if v >= 0 && v < best {
                best = v;
            }
        }
        best
    });
    for &s in &order {
        let ui = gs.unit_at[s];
        if ui < 0 {
            continue;
        }
        let u = &gs.units[ui as usize];
        let boat = is_boat(u);
        let maps = if boat { &goal_all } else { &goal_land };
        let mut best_g: Option<usize> = None;
        let mut best_d = i32::MAX;
        for (gi, m) in maps.iter().enumerate() {
            if claimed[gi] {
                continue;
            }
            let v = m[s];
            if v >= 0 {
                let score = v + goals[gi].1;
                if score < best_d {
                    best_d = score;
                    best_g = Some(gi);
                }
            }
        }
        if let Some(gi) = best_g {
            if threat && best_d > 3 {
                // under attack: only very close grabs, otherwise defend
            } else {
                claimed[gi] = true;
                roles[s] = Role::Goal(gi);
                continue;
            }
        }
        if boat {
            if !enemy_cities.is_empty() && enemy_all[s] >= 0 {
                roles[s] = Role::Enemy;
            } else if front_all[s] >= 0 {
                roles[s] = Role::Front;
            }
            continue;
        }
        // land unit
        if threat && enemy_land[s] >= 0 {
            roles[s] = Role::Enemy;
            continue;
        }
        if front_land[s] >= 0 {
            roles[s] = Role::Front;
            continue;
        }
        if naval && gs.tech[10] && !ports.is_empty() && port_land[s] >= 0 {
            roles[s] = Role::Port;
            continue;
        }
        if enemy_land[s] >= 0 {
            roles[s] = Role::Enemy;
        }
    }
    // score moves
    let mut best_score = 0.02f32;
    let mut best_id: Option<String> = None;
    for l in gs.legal.iter().filter(|l| l.kind == 1) {
        let s = l.source as usize;
        let t = l.target as usize;
        if s >= gs.n || t >= gs.n {
            continue;
        }
        let ui = gs.unit_at[s];
        if ui < 0 {
            continue;
        }
        let u = &gs.units[ui as usize];
        if u.owner != gs.me {
            continue;
        }
        let boat = is_boat(u);
        let dmap: &Vec<i32> = match roles[s] {
            Role::Goal(gi) => {
                if boat {
                    &goal_all[gi]
                } else {
                    &goal_land[gi]
                }
            }
            Role::Port => &port_land,
            Role::Front => {
                if boat {
                    &front_all
                } else {
                    &front_land
                }
            }
            Role::Enemy => {
                if boat {
                    &enemy_all
                } else {
                    &enemy_land
                }
            }
            Role::Idle => continue,
        };
        let cur = dmap[s];
        let nt = dmap[t];
        let mut score = if cur < 0 {
            if nt < 0 {
                -1.0
            } else {
                40.0 - nt as f32
            }
        } else if nt < 0 {
            -1.0
        } else {
            (cur - nt) as f32
        };
        if score < 0.0 {
            continue;
        }
        if gs.struc[t] == 1 && gs.owner[t] == gs.me {
            score += if threat { 0.8 } else { -1.3 };
        }
        if gs.terr[t] == 4 {
            score += if threat { 0.35 } else { 0.15 };
        }
        if threat && gs.terr[t] == 5 {
            score += 0.3;
        }
        if matches!(roles[s], Role::Front) {
            score += 0.22 * unknown_nbrs(gs, t) as f32 - 0.05;
        }
        if matches!(roles[s], Role::Goal(_)) && nt == 0 {
            score += 0.5;
        }
        if score > best_score {
            best_score = score;
            best_id = Some(l.id.clone());
        }
    }
    best_id
}

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout();
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
        writeln!(out, "{}", json!({ "action_id": id })).unwrap();
        out.flush().unwrap();
    }
}
