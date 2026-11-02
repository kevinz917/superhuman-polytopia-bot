// Bardur 1v1 skirmish bot: greedy per-action heuristic with map memory,
// BFS goal assignment, combat estimation, tech priorities and navy support.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn vi(v: &Value, k: &str) -> i64 { v[k].as_i64().unwrap_or(0) }
fn vf(v: &Value, k: &str) -> f64 { v[k].as_f64().unwrap_or(0.0) }

#[derive(Clone)]
struct T { known: bool, terr: i64, res: i64, struc: i64, owner: i64, road: bool }

#[derive(Clone)]
struct U { tile: usize, kind: i64, hp: f64, vet: bool, owner: i64 }

#[derive(Clone)]
struct C { tile: usize, owner: i64, level: i64, progress: i64, wall: bool, pop: i64 }

// (cost, hp, atk, def, mov, range, stiff, fortify)
fn stats(kind: i64) -> (f64, f64, f64, f64, i64, i64, bool, bool) {
    match kind {
        2 => (2., 10., 2., 2., 1, 1, false, true),
        3 => (3., 10., 2., 1., 2, 1, false, true),
        4 => (8., 10., 3.5, 1., 3, 1, false, true),
        5 => (3., 15., 1., 3., 1, 1, false, true),
        8 => (8., 10., 4., 0., 1, 3, true, false),
        9 => (3., 10., 2., 1., 1, 2, false, true),
        10 => (5., 10., 0., 1., 1, 1, true, false),
        11 => (5., 15., 3., 3., 1, 1, false, false),
        12 => (10., 40., 5., 4., 1, 1, false, false),
        15 => (2., 15., 3., 1., 1, 1, false, true),
        41 => (15., 10., 3., 2., 2, 3, true, false),
        42 => (5., 10., 2., 1., 3, 2, false, false),
        43 => (0., 10., 0., 1., 2, 2, true, false),
        44 => (5., 10., 3., 3., 3, 1, false, false),
        45 => (10., 40., 4., 4., 2, 1, true, false),
        _ => (3., 10., 2., 1., 1, 1, false, false),
    }
}

struct Bot {
    n: usize,
    player: i64,
    turn: i64,
    stars: i64,
    tech: Vec<bool>,
    tiles: Vec<T>,
    enemy_cities: Vec<(usize, i64)>, // remembered enemy city tiles
    have_port: bool,
    rng: u64,
}

const INF: i64 = 1 << 20;

impl Bot {
    fn new() -> Self {
        Bot { n: 14, player: 1, turn: 0, stars: 0, tech: vec![false; 128], tiles: Vec::new(),
              enemy_cities: Vec::new(), have_port: false, rng: 0x2545F4914F6CDD1D }
    }

    fn update(&mut self, obs: &Value) {
        let vs = &obs["visible_state"];
        let n = vi(vs, "board_size") as usize;
        if self.tiles.is_empty() {
            self.n = n;
            self.tiles = vec![T { known: false, terr: 3, res: 0, struc: 0, owner: 0, road: false }; n * n];
        }
        self.player = vi(obs, "player");
        self.turn = vi(vs, "turn");
        self.stars = vf(vs, "stars") as i64;
        self.tech = vec![false; 128];
        if let Some(arr) = vs["technology"].as_array() {
            for v in arr.iter() {
                let id = v.as_i64().unwrap_or(-1);
                if id >= 0 && (id as usize) < 128 { self.tech[id as usize] = true; }
            }
        }
        if let Some(arr) = vs["tiles"].as_array() {
            for t in arr {
                let idx = vi(t, "tile") as usize;
                if idx >= self.tiles.len() { continue; }
                if t["known"].as_i64().unwrap_or(0) == 1 || t["known"].as_bool() == Some(true) {
                    let old = &mut self.tiles[idx];
                    old.known = true;
                    if let Some(x) = t.get("terrain") { old.terr = x.as_i64().unwrap_or(old.terr); }
                    if let Some(x) = t.get("resource") { old.res = x.as_i64().unwrap_or(0); }
                    if let Some(x) = t.get("structure") { old.struc = x.as_i64().unwrap_or(0); }
                    if let Some(x) = t.get("owner") { old.owner = x.as_i64().unwrap_or(0); }
                    old.road = t.get("road").map(|r| r.as_i64().unwrap_or(0) > 0).unwrap_or(old.road);
                    if old.struc == 8 && old.owner == self.player { self.have_port = true; }
                }
            }
        }
    }

    fn road_at(&self, i: usize) -> bool {
        let t = &self.tiles[i];
        t.road || t.struc == 1 || t.struc == 47 || (t.struc == 8)
    }

    fn passable(&self, i: usize, aquatic: bool) -> bool {
        let t = &self.tiles[i];
        if !t.known { return !aquatic; } // assume land for unknown
        let water = t.terr == 1 || t.terr == 2;
        if aquatic { return true; }
        if water { return t.struc == 8 && t.owner == self.player; }
        if t.terr == 4 { return self.tech[20]; }
        true
    }

    fn expandable(&self, i: usize, aquatic: bool) -> bool {
        let t = &self.tiles[i];
        if !t.known { return true; }
        let water = t.terr == 1 || t.terr == 2;
        if aquatic { return water; } // boats stop on land (disembark)
        if (t.terr == 5 || t.terr == 4) && !self.road_at(i) { return false; }
        true
    }

    fn dist_map(&self, src: usize, aquatic: bool) -> Vec<i64> {
        let n = self.n;
        let total = n * n;
        let mut d = vec![INF; total];
        d[src] = 0;
        for _ in 0..16 {
            let mut changed = false;
            for y in 0..n {
                for x in 0..n {
                    let i = y * n + x;
                    if d[i] >= INF { continue; }
                    if !self.expandable(i, aquatic) && i != src { continue; }
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            if dx == 0 && dy == 0 { continue; }
                            let (ny, nx) = ((y as i64 + dy) as usize, (x as i64 + dx) as usize);
                            if ny >= n || nx >= n { continue; }
                            let j = ny * n + nx;
                            if !self.passable(j, aquatic) { continue; }
                            let cost = if self.road_at(i) && self.road_at(j) { 1 } else { 2 };
                            if d[i] + cost < d[j] { d[j] = d[i] + cost; changed = true; }
                        }
                    }
                }
            }
            if !changed { break; }
        }
        d
    }

    fn cheb(&self, a: usize, b: usize) -> i64 {
        let n = self.n as i64;
        let (ax, ay) = ((a % self.n) as i64, (a / self.n) as i64);
        let (bx, by) = ((b % self.n) as i64, (b / self.n) as i64);
        let _ = n;
        (ax - bx).abs().max((ay - by).abs())
    }
}

fn combat(atk: f64, a_hp: f64, a_max: f64, def: f64, d_hp: f64, d_max: f64, bonus: f64) -> (f64, f64) {
    let af = atk * (a_hp / a_max.maxi(1.0));
    let df = def * (d_hp / d_max.maxi(1.0)) * bonus;
    let total = af + df;
    if total <= 0.0 { return (0.0, 0.0); }
    let dmg = ((af / total) * atk * 4.5 + 0.000001 + 0.5).floor();
    let ret = if d_hp > dmg { ((df / total) * def * 4.5 + 0.000001 + 0.5).floor() } else { 0.0 };
    (dmg, ret)
}

trait Maxi { fn maxi(self, o: f64) -> f64; }
impl Maxi for f64 { fn maxi(self, o: f64) -> f64 { if self > o { self } else { o } } }

fn choose(bot: &mut Bot, obs: &Value) -> String {
    bot.update(obs);
    let vs = &obs["visible_state"];
    let legal = match obs["legal_actions"].as_array() { Some(a) => a.clone(), None => return String::from("0") };
    if legal.is_empty() { return String::from("0"); }

    // --- parse world ---
    let mut my_units: Vec<U> = Vec::new();
    let mut en_units: Vec<U> = Vec::new();
    if let Some(arr) = vs["units"].as_array() {
        for u in arr {
            let un = U { tile: vi(u, "tile") as usize, kind: vi(u, "kind"), hp: vf(u, "health"),
                         vet: u["veteran"].as_i64().unwrap_or(0) == 1, owner: vi(u, "owner") };
            if un.owner == 0 { continue; }
            if un.owner == bot.player { my_units.push(un); } else { en_units.push(un); }
        }
    }
    let mut my_cities: Vec<C> = Vec::new();
    let mut en_city_tiles: Vec<usize> = Vec::new();
    if let Some(arr) = vs["cities"].as_array() {
        for c in arr {
            let owner = vi(c, "owner");
            if owner == 0 { continue; }
            let wall = c.get("rewards").and_then(|r| r.as_array()).map(|r| r.get(1).map(|x| x.as_i64().unwrap_or(0) > 0).unwrap_or(false)).unwrap_or(false);
            let cc = C { tile: vi(c, "tile") as usize, owner, level: c.get("level").and_then(|x| x.as_i64()).unwrap_or(1),
                         progress: c.get("progress").and_then(|x| x.as_i64()).unwrap_or(0), wall, pop: 0 };
            if owner == bot.player { my_cities.push(cc); } else { en_city_tiles.push(cc.tile); }
        }
    }
    for &t in &en_city_tiles {
        if !bot.enemy_cities.iter().any(|&(x, _)| x == t) { bot.enemy_cities.push((t, bot.turn)); }
    }
    bot.enemy_cities.retain(|&(x, _)| !my_cities.iter().any(|c| c.tile == x));
    let enemy_known = !bot.enemy_cities.is_empty() || !en_units.is_empty();

    let my_city_tiles: Vec<usize> = my_cities.iter().map(|c| c.tile).collect();
    let income: i64 = my_cities.iter().map(|c| c.level.max(1)).sum();

    // --- exploration / map stats ---
    let total = bot.n * bot.n;
    let mut known_cnt = 0; let mut water_cnt = 0;
    let mut neutral_villages: Vec<usize> = Vec::new();
    let mut ruins: Vec<usize> = Vec::new();
    let mut frontier: Vec<usize> = Vec::new();
    for (idx, t) in bot.tiles.iter().enumerate() {
        if !t.known { continue; }
        known_cnt += 1;
        if t.terr == 1 || t.terr == 2 { water_cnt += 1; }
        if t.struc == 1 && t.owner == 0 { neutral_villages.push(idx); }
        if t.struc == 2 { ruins.push(idx); }
    }
    let known_frac = known_cnt as f64 / total as f64;
    let water_frac = water_cnt as f64 / known_cnt.max(1) as f64;
    for idx in 0..total {
        if bot.tiles[idx].known { continue; }
        // adjacent to a known tile => frontier
        let (x, y) = (idx % bot.n, idx / bot.n);
        let mut adj_known = false;
        for (dx, dy) in [(-1i64,0i64),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)] {
            let nx = x as i64 + dx; let ny = y as i64 + dy;
            if nx >= 0 && ny >= 0 && (nx as usize) < bot.n && (ny as usize) < bot.n {
                if bot.tiles[ny as usize * bot.n + nx as usize].known { adj_known = true; }
            }
        }
        if adj_known { frontier.push(idx); }
    }

    // --- threat & aggression ---
    let mut threat: i64 = 0;
    for e in &en_units {
        for &c in &my_city_tiles {
            if bot.cheb(e.tile, c) <= 3 { threat += 1; }
        }
    }
    let my_strength: f64 = my_units.iter().map(|u| stats(u.kind).0).sum();
    let en_strength: f64 = en_units.iter().map(|u| stats(u.kind).0).sum();
    let mut aggressive = bot.turn > 50 || my_units.len() as i64 >= en_units.len() as i64 + 2
        || (bot.turn > 25 && my_cities.len() >= en_city_tiles.len() + 2 && enemy_known);
    if my_strength < en_strength * 0.8 { aggressive = false; }
    let outgunned = enemy_known && en_strength > my_strength * 1.35;

    // --- navy need ---
    let land_d = bot.dist_map(my_city_tiles.first().copied().unwrap_or(0), false);
    let reachable_villages = neutral_villages.iter().filter(|&&v| land_d[v] < INF).count();
    let need_navy = water_frac > 0.30
        || (bot.turn > 30 && reachable_villages == 0 && (frontier.len() > 10 || !enemy_known));

    // --- goals ---
    let mut goals: Vec<(usize, f64)> = Vec::new();
    let village_value = if outgunned { 210.0 } else { 430.0 } - bot.turn as f64 * 1.0;
    for &v in &neutral_villages { goals.push((v, village_value.max(170.0))); }
    for &r in &ruins { goals.push((r, 330.0)); }
    if aggressive {
        for &t in &en_city_tiles { goals.push((t, 340.0)); }
        for e in &en_units { goals.push((e.tile, 240.0)); }
        for &(t, _) in &bot.enemy_cities { if !en_city_tiles.contains(&t) { goals.push((t, 300.0)); } }
    } else if enemy_known {
        for e in &en_units {
            let near_home = my_city_tiles.iter().any(|&c| bot.cheb(e.tile, c) <= 6);
            if near_home { goals.push((e.tile, 460.0)); }
        }
    }
    // home garrison attraction
    if let Some(&cap) = my_city_tiles.first() { goals.push((cap, 235.0)); }
    if outgunned { for &c in &my_city_tiles { goals.push((c, 520.0)); } }
    let explore_value = if known_frac < 0.5 { 185.0 } else if known_frac < 0.85 { 140.0 } else { 60.0 };
    let cap0 = my_city_tiles.first().copied().unwrap_or(0);
    for &f in &frontier {
        let v = explore_value - bot.cheb(f, cap0) as f64 * 1.6;
        goals.push((f, v));
    }
    if threat > 0 {
        for &c in &my_city_tiles {
            let near_en = en_units.iter().any(|e| bot.cheb(e.tile, c) <= 6);
            if near_en { goals.push((c, 600.0)); }
        }
    }
    if need_navy && !bot.have_port {
        // head to coast to enable port area scouting: target known shallow water in our territory edge
    }

    // --- per-unit distance maps & goals ---
    let mut dists: std::collections::HashMap<usize, Vec<i64>> = std::collections::HashMap::new();
    let mut unit_goal: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut srcs: Vec<usize> = Vec::new();
    for a in &legal {
        if vi(&a["command"], "kind") == 1 {
            let s = vi(&a["command"], "source") as usize;
            if !srcs.contains(&s) { srcs.push(s); }
        }
    }
    let mut dists: std::collections::HashMap<usize, Vec<i64>> = std::collections::HashMap::new();
    let mut unit_goal: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut srcs: Vec<usize> = Vec::new();
    for a in &legal {
        if vi(&a["command"], "kind") == 1 {
            let s = vi(&a["command"], "source") as usize;
            if !srcs.contains(&s) { srcs.push(s); }
        }
    }
    // cities where we are locally outnumbered -> hunker down
    let mut hunker: Vec<usize> = Vec::new();
    for &c in &my_city_tiles {
        let ev: f64 = en_units.iter().filter(|e| bot.cheb(e.tile, c) <= 3).map(|e| stats(e.kind).0).sum();
        let mv: f64 = my_units.iter().filter(|u| bot.cheb(u.tile, c) <= 3).map(|u| stats(u.kind).0).sum();
        if ev >= 6.0 || ev > mv * 2.0 + 2.0 { hunker.push(c); }
    }
    let mut claimed_village: Vec<usize> = Vec::new();
    let mut claimed_frontier: Vec<usize> = Vec::new();
    let mut claimed_enemy: Vec<usize> = Vec::new();
    for (ui, &s) in srcs.iter().enumerate() {
        let un = my_units.iter().find(|u| u.tile == s);
        let aquatic = match un { Some(u) => matches!(u.kind, 39 | 40 | 41 | 42 | 43 | 44 | 45), None => false };
        let d = bot.dist_map(s, aquatic);
        // this unit's exploration sector (8 sectors around capital)
        let sect = (ui % 8) as i64;
        let (cx, cy) = ((cap0 % bot.n) as i64, (cap0 / bot.n) as i64);
        let sector_of = |g: usize| -> i64 {
            let (gx, gy) = ((g % bot.n) as i64, (g / bot.n) as i64);
            let (dx, dy) = (gx - cx, cy - gy); // dy up positive
            let mut ang = (dy as f64).atan2(dx as f64);
            if ang < 0.0 { ang += std::f64::consts::TAU; }
            ((ang / (std::f64::consts::TAU / 8.0)).round() as i64) % 8
        };
        let mut best_v = -1e9; let mut best_g = s;
        for &(g, v) in &goals {
            if d[g] >= INF { continue; }
            let mut v = v;
            let t = &bot.tiles[g];
            if !t.known {
                if sector_of(g) == sect { v += 55.0; }
                v += ((g as u64).wrapping_mul(2654435761) % 97) as f64 * 0.05;
            }
            if t.struc == 1 || t.struc == 2 {
                if claimed_village.contains(&g) { continue; }
            } else if !t.known {
                if claimed_frontier.iter().any(|&f| bot.cheb(f, g) <= 3) { continue; }
            } else if t.owner == bot.player {
                // own city (garrison/defense): allow up to 2 claimants
                if claimed_enemy.iter().filter(|&&x| x == g).count() >= 2 { continue; }
            } else {
                if claimed_enemy.iter().filter(|&&x| x == g).count() >= 3 { continue; }
            }
            let score = v - d[g] as f64 * 8.0;
            if score > best_v { best_v = score; best_g = g; }
        }
        for &c in &hunker {
            if d[c] < INF {
                let score = 700.0 - d[c] as f64 * 8.0;
                if score > best_v { best_v = score; best_g = c; }
            }
        }
        if best_g != s {
            let t = &bot.tiles[best_g];
            if t.struc == 1 || t.struc == 2 { claimed_village.push(best_g); }
            else if !t.known { claimed_frontier.push(best_g); }
            else { claimed_enemy.push(best_g); }
        }
        if std::env::var("DBG").is_ok() {
            eprintln!("turn {} unit {} goal {} bestv {:.0}", bot.turn, s, best_g, best_v);
        }
        unit_goal.insert(s, best_g);
        dists.insert(s, d);
    }

    // distance maps FROM each distinct goal (for move progress scoring)
    let mut goal_dists: std::collections::HashMap<usize, Vec<i64>> = std::collections::HashMap::new();
    {
        let mut gs: Vec<usize> = unit_goal.values().copied().filter(|&g| dists.values().any(|_| true)).collect();
        gs.retain(|&g| g != 0 || true);
        gs.sort(); gs.dedup();
        for g in gs {
            if goal_dists.contains_key(&g) { continue; }
            let aquatic = my_units.iter().find(|u| u.tile == g).map(|u| matches!(u.kind, 39 | 40 | 41 | 42 | 43 | 44 | 45)).unwrap_or(false);
            goal_dists.insert(g, bot.dist_map(g, aquatic));
        }
    }

    let end_turn_score = 20.0;
    let mut best_score = -1e9;
    let mut best_id = String::new();
    let mut my_unit_count = my_units.len() as i64;

    for a in &legal {
        let cmd = &a["command"];
        let kind = vi(cmd, "kind");
        let param = vi(cmd, "parameter");
        let src = vi(cmd, "source");
        let tgt = vi(cmd, "target");
        let mut sc = 0.0;

        match kind {
            9 => { // city reward choice
                let threatened = threat > 0;
                sc = match param {
                    4 => if known_frac < 0.9 { 9300.0 } else { 8700.0 }, // explorer
                    3 => 9000.0, // workshop
                    7 => 9100.0, // resources +5 stars
                    1 => if threatened || outgunned { 9450.0 } else { 8600.0 }, // wall
                    8 => 9000.0, // population +3
                    5 => 8950.0, // border growth
                    6 => if bot.turn > 8 || threatened { 9200.0 } else { 8800.0 }, // giant
                    2 => 8700.0, // park
                    _ => 8900.0,
                };
            }
            8 => { sc = 8000.0; } // capture village/ruin
            3 => {
                match param {
                    16 => sc = 7000.0, // promote
                    7 => { // recover
                        if let Some(u) = my_units.iter().find(|u| u.tile == src as usize) {
                            let mx = stats(u.kind).1 + if u.vet { 5.0 } else { 0.0 };
                            let enemies_near = en_units.iter().any(|e| bot.cheb(e.tile, u.tile) <= 2);
                            let on_my_city = my_city_tiles.contains(&(src as usize));
                            if on_my_city && enemies_near { sc = 380.0 + (mx - u.hp) * 12.0; }
                            else if enemies_near { sc = 60.0; }
                            else { sc = 150.0 + (mx - u.hp) * 10.0; }
                        }
                    }
                    9 => sc = 200.0, // heal others
                    _ => sc = -100.0,
                }
            }
            2 => { // attack
                let au = my_units.iter().find(|u| u.tile == src as usize);
                let du = en_units.iter().find(|u| u.tile == tgt as usize);
                if let (Some(au), Some(du)) = (au, du) {
                    let (acost, ahp, aatk, _, _, _, _, afort) = stats(au.kind);
                    let (_, dhp, _, ddef, _, drange, dstiff, dfort) = stats(du.kind);
                    let amax = ahp + if au.vet { 5.0 } else { 0.0 };
                    let dmax = dhp + if du.vet { 5.0 } else { 0.0 };
                    let t = &bot.tiles[tgt as usize];
                    let mut bonus: f64 = 1.0;
                    let is_city = t.struc == 1 || t.struc == 47;
                    if is_city && dfort { bonus = if du.owner == t.owner { 1.5 } else { 1.0 }; }
                    if t.terr == 5 || t.terr == 4 { bonus *= 1.5; }
                    if t.terr == 1 || t.terr == 2 { bonus *= 1.5; }
                    let (dmg, ret) = combat(aatk, au.hp, amax, ddef, du.hp, dmax, bonus);
                    let kill = dmg >= du.hp;
                    let mut v = dmg * 40.0;
                    if kill { v += 2500.0 + acost * 260.0; }
                    let can_retal = !kill && !dstiff && bot.cheb(src as usize, tgt as usize) <= drange;
                    if can_retal {
                        v -= ret * 34.0;
                        if ret >= au.hp { v -= 700.0 + acost * 130.0; }
                    }
                    // tactical bonuses
                    if my_city_tiles.iter().any(|&c| bot.cheb(tgt as usize, c) <= 2) { v += 700.0; }
                    if en_city_tiles.contains(&(tgt as usize)) { v += 500.0; }
                    if du.kind == 12 { v += 300.0; } // chip the giant
                    let near_my_city = my_city_tiles.iter().any(|&c| bot.cheb(tgt as usize, c) <= 3);
                    let safe = kill || (can_retal && ret <= 2.0 && dmg >= 3.0) || !can_retal || au.hp >= 8.0;
                    if !near_my_city && !safe { v *= 0.3; }
                    else if !kill && !near_my_city && my_units.len() < 5 { v *= 0.7; }
                    if en_strength > my_strength * 1.4 && !kill && !near_my_city { v *= 0.2; }
                    sc = v;
                }
            }
            5 => { // harvest
                let in_my_land = bot.tiles[tgt as usize].owner == bot.player;
                let mut v = if bot.turn < 15 { 980.0 } else if bot.turn < 60 { 520.0 } else { 330.0 };
                if let Some(c) = my_cities.iter().find(|c| c.tile != 0 && city_owns(bot, c.tile, tgt as usize)) {
                    if c.progress + 1 >= c.level + 1 { v += 200.0; }
                    if c.level >= 5 { v -= 250.0; }
                }
                if !in_my_land { v -= 500.0; }
                if bot.stars < 4 && my_unit_count < 3 { v -= 150.0; }
                sc = v;
            }
            6 => { // build
                let cost = match param {
                    3 => 3.0, 5 => 5.0, 6 => 5.0, 8 => 7.0, 12 => 3.0, 13 => 5.0, 21 => 5.0, 22 => 5.0, 50 => 5.0, _ => 5.0,
                };
                let mut v = match param {
                    12 => if bot.turn < 25 { 700.0 } else { 520.0 }, // lumber hut
                    5 => 560.0,         // farm
                    21 => 560.0,        // mine
                    13 => 540.0,        // sawmill
                    6 => 520.0,         // windmill
                    22 => 520.0,        // forge
                    50 => 430.0,        // market
                    8 => if need_navy && !bot.have_port { 880.0 } else { 100.0 }, // port
                    3 => 120.0,         // road
                    _ => 200.0,
                };
                if (bot.stars as f64) < cost + 3.0 { v -= 350.0; }
                if bot.turn < 6 { v -= 250.0; }
                sc = v;
            }
            4 => { // train/summon
                let (ucost, _, _, _, _, _, _, _) = stats(param);
                let kw = match param {
                    2 => 1.0, 3 => 1.15, 11 => 1.3, 4 => 1.35, 9 => 0.95, 5 => 0.55,
                    8 => 0.7, 42 => if need_navy { 1.25 } else { 0.1 },
                    44 => if need_navy { 1.1 } else { 0.1 },
                    _ => 0.5,
                };
                let mut target_units = (5 + my_cities.len() as i64 * 2).max(6);
                if aggressive { target_units += 3; }
                if threat > 0 { target_units += 2; }
                target_units = target_units.min(18);
                let mut v = 0.0;
                let swarm = en_units.iter().filter(|e| bot.cheb(e.tile, src as usize) <= 2).count() as f64;
                if swarm >= 3.0 { /* spawning into certain death */ }
                if my_unit_count < target_units {
                    v = 800.0 + (target_units - my_unit_count) as f64 * 45.0 + threat as f64 * 150.0;
                    if my_unit_count < 6 { v = v.max(980.0); }
                    v *= kw;
                    let spawn_walled = my_cities.iter().any(|c| c.tile == src as usize && c.wall);
                    if swarm >= 3.0 && !spawn_walled { v *= 0.15; }
                    if bot.stars as f64 > ucost + 6.0 { v += 160.0; }
                }
                // navy transport: prefer embarkable spot
                sc = v;
            }
            7 => { // research
                sc = tech_score(param, bot.turn, water_frac, need_navy, my_cities.len());
                let tier = tech_tier(param);
                let cost = 4 + my_cities.len() as i64 * tier;
                if bot.stars < cost + 2 { sc *= 0.55; }
                if bot.turn < 6 && my_unit_count < 2 { sc *= 0.35; }
                else if bot.stars < cost + 8 && bot.turn < 20 { sc *= 0.45; }
            }
            1 => { // move
                let s = src as usize;
                let opt = dists.get(&s);
                if opt.is_none() { sc = 15.0; } else {
                    let g = *unit_goal.get(&s).unwrap_or(&s);
                    let mut v = 140.0;
                    if g != s {
                        let gd = goal_dists.get(&g);
                        let cheb_to = |t: usize| bot.cheb(t, g) as f64;
                        let (before, after) = match gd {
                            Some(gd) => (gd[s] as f64, gd.get(tgt as usize).copied().unwrap_or(INF) as f64),
                            None => (cheb_to(s), cheb_to(tgt as usize)),
                        };
                        if after < INF as f64 { v += (before - after) * 26.0; }
                    }
                    let t = &bot.tiles[tgt as usize];
                    if t.struc == 1 && t.owner == 0 { v += 260.0; }
                    if t.struc == 2 { v += 240.0; }
                    if !t.known { v += 30.0; }
                    let enemies_near = en_units.iter().any(|e| bot.cheb(e.tile, tgt as usize) <= 2);
                    if enemies_near && (t.terr == 5 || t.terr == 4) { v += 40.0; }
                    if my_city_tiles.contains(&(tgt as usize)) && threat == 0 { v -= 30.0; }
                    if hunker.contains(&(src as usize)) { v -= 250.0; }
                    if bot.have_port && t.struc == 8 && t.owner == bot.player && need_navy { v += 420.0; }
                    if g == s { v = 15.0; }
                    sc = v;
                }
            }
            10 => { sc = end_turn_score; }
            11 => { sc = -1000.0; }
            _ => { sc = 0.0; }
        }

        // tiny deterministic tie-break noise
        let aid = a["action_id"].as_str().unwrap_or("").to_string();
        let tb = (aid.len() as f64) * 0.001 + (param as f64) * 0.0001;
        let total_score = sc + tb;
        if total_score > best_score { best_score = total_score; best_id = aid; }
    }

    if best_id.is_empty() {
        let a = &legal[0];
        my_unit_count += 0;
        return a["action_id"].as_str().unwrap_or("0").to_string();
    }
    best_id
}

fn city_owns(bot: &Bot, city_tile: usize, tile: usize) -> bool {
    // approximate: tile belongs to city if within Chebyshev 1 of city tile and same owner region
    bot.cheb(city_tile, tile) <= 1 && bot.tiles[tile].owner == bot.player
}

fn tech_tier(tech: i64) -> i64 {
    match tech {
        0 => 0,
        1 | 6 | 10 | 15 | 20 | 25 | 26 | 28 | 30 | 31 | 32 | 35 | 36 | 37 | 41 | 42 | 44 | 48 | 49 | 120 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 121 => 2,
        _ => 3,
    }
}

fn tech_score(tech: i64, turn: i64, water_frac: f64, need_navy: bool, cities: usize) -> f64 {
    let decay = (1.0 - turn as f64 / 160.0).max(0.25);
    let base = match tech {
        6 => if turn < 25 { 950.0 } else { 450.0 },   // Organization
        20 => if turn < 30 { 880.0 } else { 550.0 },  // Climbing
        1 => if turn < 35 { 920.0 } else { 650.0 },   // Riding
        8 => 950.0,                                     // Farming
        16 => if turn >= 3 { 1180.0 } else { 600.0 },  // Forestry
        18 => 850.0,                                    // Archery
        23 => 820.0,                                    // Mining
        24 => 920.0,                                    // Smithery
        17 => if turn < 55 { 650.0 } else { 300.0 },    // Mathematics
        7 => 460.0,                                     // Strategy
        10 => if water_frac > 0.28 { 980.0 } else { 330.0 }, // Fishing
        13 => if need_navy { 1180.0 } else { 300.0 },   // Sailing
        14 => if turn > 35 && water_frac > 0.28 { 620.0 } else { 200.0 }, // Navigation
        2 => if turn > 28 { 720.0 } else { 250.0 },     // FreeSpirit
        3 => if turn > 28 { 980.0 } else { 300.0 },     // Chivalry
        21 => 240.0,                                    // Meditation
        22 => if turn > 12 && turn < 70 && cities >= 3 { 600.0 } else { 250.0 }, // Philosophy
        9 => 420.0,                                     // Construction
        4 => 220.0,                                     // Roads
        5 => if turn > 40 { 460.0 } else { 200.0 },     // Trade
        44 => if need_navy && turn > 30 { 520.0 } else { 150.0 }, // Ramming
        _ => 250.0,
    };
    base * decay
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut bot = Bot::new();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let action = choose(&mut bot, &obs);
        writeln!(stdout, "{}", json!({"action_id": action})).unwrap();
        stdout.flush().unwrap();
    }
}
