use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Clone, Default)]
struct Tile { known: bool, terrain: i64, owner: i64, structure: i64 }
#[derive(Clone, Default)]
struct Unit { owner: i64, kind: i64, hp: f64, veteran: bool }

struct G {
    size: usize, n: usize, me: i64, stars: i64, tech: [bool; 128],
    tiles: Vec<Tile>, units: Vec<Option<Unit>>, cities: Vec<(usize, i64, i64)>, capital: Vec<i64>,
}

const INF: i32 = 1_000_000;

fn ustats(kind: i64) -> (f64, f64, f64, i32, i64) {
    match kind {
        2 => (2., 2., 10., 1, 2), 3 => (2., 1., 10., 1, 3), 4 => (3.5, 1., 10., 1, 8), 5 => (1., 3., 15., 1, 3),
        8 => (4., 0., 10., 3, 8), 9 => (2., 1., 10., 2, 3), 10 => (0., 1., 10., 1, 5), 11 => (3., 3., 15., 1, 5),
        12 => (5., 4., 40., 1, 10), 37 => (2., 2., 10., 1, 2), 38 => (0., 0.5, 5., 1, 8), 39 => (0., 0.5, 5., 1, 2),
        40 => (2., 2., 10., 1, 2), 41 => (3., 2., 10., 3, 15), 42 => (2., 1., 10., 2, 5), 43 => (0., 1., 10., 2, 0),
        44 => (3., 3., 10., 1, 5), 45 => (4., 4., 40., 1, 10), _ => (2., 2., 10., 1, 2),
    }
}
fn is_naval(kind: i64) -> bool { matches!(kind, 39 | 40 | 41 | 42 | 43 | 44 | 45) }
fn stiff(kind: i64) -> bool { matches!(kind, 8 | 10 | 38 | 39 | 41 | 43) }
fn fortify(kind: i64) -> bool { matches!(kind, 2 | 3 | 4 | 5 | 9 | 15 | 20) }

fn combat(aa: f64, ah: f64, am: f64, dd: f64, dh: f64, dm: f64, db: f64) -> (f64, f64) {
    let af = aa * ah / am.max(1.0);
    let df = dd * dh / dm.max(1.0) * db;
    let tot = af + df;
    if tot <= 0.0 { return (0.0, 0.0); }
    let atk = (af / tot * aa * 4.5 + 0.5).floor();
    let ret = if dh > atk { (df / tot * dd * 4.5 + 0.5).floor() } else { 0.0 };
    (atk, ret)
}

impl G {
    fn parse(obs: &Value) -> G {
        let vs = &obs["visible_state"];
        let size = vs["board_size"].as_u64().unwrap_or(14) as usize;
        let n = size * size;
        let me = obs["player"].as_i64().unwrap_or(1);
        let stars = vs["stars"].as_i64().unwrap_or(0);
        let mut tech = [false; 128];
        if let Some(ts) = vs["technology"].as_array() {
            for t in ts { if let Some(i) = t.as_i64() { if i >= 0 && i < 128 { tech[i as usize] = true; } } }
        }
        let mut tiles = vec![Tile::default(); n];
        let mut capital = vec![0i64; n];
        if let Some(ta) = vs["tiles"].as_array() {
            for t in ta {
                let i = t["tile"].as_u64().unwrap_or(0) as usize;
                if i >= n { continue; }
                let known = t["known"].as_bool().unwrap_or(false);
                tiles[i] = Tile {
                    known,
                    terrain: t["terrain"].as_i64().unwrap_or(0),
                    owner: t["owner"].as_i64().unwrap_or(0),
                    structure: t["structure"].as_i64().unwrap_or(0),
                };
                capital[i] = t["capital"].as_i64().unwrap_or(0);
            }
        }
        let mut units = vec![None; n];
        if let Some(ua) = vs["units"].as_array() {
            for u in ua {
                let i = u["tile"].as_u64().unwrap_or(0) as usize;
                if i >= n { continue; }
                units[i] = Some(Unit {
                    owner: u["owner"].as_i64().unwrap_or(0),
                    kind: u["kind"].as_i64().unwrap_or(2),
                    hp: u["health"].as_f64().unwrap_or(10.0),
                    veteran: u["veteran"].as_bool().unwrap_or(false),
                });
            }
        }
        let mut cities = Vec::new();
        if let Some(ca) = vs["cities"].as_array() {
            for c in ca {
                let i = c["tile"].as_u64().unwrap_or(0) as usize;
                if i >= n { continue; }
                cities.push((i, c["owner"].as_i64().unwrap_or(0), c["level"].as_i64().unwrap_or(1)));
            }
        }
        G { size, n, me, stars, tech, tiles, units, cities, capital }
    }

    fn d(&self, a: usize, b: usize) -> i32 {
        let s = self.size as i32;
        let (ax, ay) = ((a as i32) % s, (a as i32) / s);
        let (bx, by) = ((b as i32) % s, (b as i32) / s);
        (ax - bx).abs().max((ay - by).abs())
    }
    fn neighbors(&self, i: usize) -> Vec<usize> {
        let s = self.size as i32;
        let (x, y) = ((i as i32) % s, (i as i32) / s);
        let mut v = Vec::with_capacity(8);
        for dy in -1..=1 { for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && nx < s && ny >= 0 && ny < s { v.push((ny * s + nx) as usize); }
        }}
        v
    }
    fn enemy_unit_at(&self, i: usize) -> bool {
        matches!(&self.units[i], Some(u) if u.owner != self.me)
    }
    fn own_unit_at(&self, i: usize) -> bool {
        matches!(&self.units[i], Some(u) if u.owner == self.me)
    }
    fn land_pass(&self, i: usize) -> bool {
        let t = &self.tiles[i];
        if !t.known { return true; }
        if self.enemy_unit_at(i) { return false; }
        t.terrain == 3 || t.terrain == 5 || (t.terrain == 4 && self.tech[20]) || t.structure == 48
    }
    fn water_pass(&self, i: usize) -> bool {
        let t = &self.tiles[i];
        if !t.known { return true; }
        if self.enemy_unit_at(i) { return false; }
        t.terrain == 1 || (t.terrain == 2 && self.tech[13])
    }
    fn is_land(&self, i: usize) -> bool {
        let t = &self.tiles[i];
        t.known && t.terrain >= 3
    }
    // BFS distance from goal `start` for a unit type.
    fn bfs(&self, start: usize, naval: bool) -> Vec<i32> {
        let mut dist = vec![INF; self.n];
        let st = &self.tiles[start];
        let terrain_ok = !st.known || if naval { st.terrain >= 1 } else { st.terrain == 3 || st.terrain == 5 || (st.terrain == 4 && self.tech[20]) };
        if !terrain_ok { return dist; }
        let mut q = std::collections::VecDeque::new();
        dist[start] = 0;
        q.push_back(start);
        while let Some(x) = q.pop_front() {
            let dx = dist[x];
            // expansion allowed?
            if x != start {
                let ok = if naval { self.water_pass(x) && !self.is_land(x) } else { self.land_pass(x) };
                if !ok { continue; }
            }
            for y in self.neighbors(x) {
                if dist[y] != INF { continue; }
                let enter = if naval {
                    self.water_pass(y) || (self.is_land(y) && !self.enemy_unit_at(y))
                } else {
                    self.land_pass(y)
                };
                if enter {
                    dist[y] = dx + 1;
                    q.push_back(y);
                }
            }
        }
        dist
    }
    fn unknown_around(&self, t: usize) -> i32 {
        let r = if self.tiles[t].known && self.tiles[t].terrain == 4 { 2 } else { 1 };
        let mut c = 0;
        for i in 0..self.n {
            if self.d(t, i) <= r && !self.tiles[i].known { c += 1; }
        }
        c
    }
    fn def_bonus(&self, t: usize, u: &Unit) -> f64 {
        let tile = &self.tiles[t];
        if tile.terrain == 4 { return 1.5; }
        if tile.terrain == 5 { return if u.owner == self.me { if self.tech[18] { 1.5 } else { 1.0 } } else { 1.3 }; }
        if tile.terrain <= 2 { return 1.0; }
        if tile.structure == 1 && tile.owner == u.owner && fortify(u.kind) { return 1.5; }
        1.0
    }
}

fn choose(obs: &Value) -> String {
    let legal = match obs["legal_actions"].as_array() { Some(l) if !l.is_empty() => l, _ => return "0".into() };
    let id = |a: &Value| a["action_id"].as_str().unwrap_or("0").to_owned();
    let cmd = |a: &Value, k: &str| a["command"][k].as_i64().unwrap_or(-1);
    let g = G::parse(obs);
    let n = g.n;

    // 1. mandatory city reward
    let rewards: Vec<&Value> = legal.iter().filter(|a| cmd(a, "kind") == 9).collect();
    if !rewards.is_empty() {
        let enemy_seen = (0..n).any(|i| g.enemy_unit_at(i));
        let city = cmd(rewards[0], "target");
        let is_cap = city >= 0 && (city as usize) < n && g.capital[city as usize] == g.me;
        let wall_first = is_cap || enemy_seen;
        let prefs: Vec<i64> = if wall_first { vec![3, 1, 8, 6, 4, 7, 5, 2] } else { vec![3, 7, 8, 6, 4, 1, 5, 2] };
        for pref in prefs {
            if let Some(a) = rewards.iter().find(|a| cmd(a, "parameter") == pref) { return id(a); }
        }
        return id(rewards[0]);
    }
    // 2. promote
    if let Some(a) = legal.iter().find(|a| cmd(a, "kind") == 3 && cmd(a, "parameter") == 16) { return id(a); }
    // 3. capture
    if let Some(a) = legal.iter().find(|a| cmd(a, "kind") == 8) { return id(a); }

    // counts
    let my_units: Vec<usize> = (0..n).filter(|&i| g.own_unit_at(i)).collect();
    let enemy_units: Vec<usize> = (0..n).filter(|&i| g.enemy_unit_at(i)).collect();
    let my_cities: Vec<(usize, i64)> = g.cities.iter().filter(|c| c.1 == g.me).map(|c| (c.0, c.2)).collect();
    let ncity = my_cities.len().max(1) as i64;
    let nunits = my_units.len() as i64;
    let known_tiles = g.tiles.iter().filter(|t| t.known).count().max(1);
    let water_tiles = g.tiles.iter().filter(|t| t.known && (t.terrain == 1 || t.terrain == 2)).count();
    let water_frac = water_tiles as f64 / known_tiles as f64;
    let ocean_seen = g.tiles.iter().any(|t| t.known && t.terrain == 2);
    let aggressive = nunits as f64 >= 1.5 * enemy_units.len() as f64 + 3.0 && nunits >= 6;
    let enemy_seen = !enemy_units.is_empty();
    let ntech = g.tech.iter().filter(|&&t| t).count() as i64;

    // 4. attack
    let mut best_attack: Option<(f64, String)> = None;
    for a in legal.iter().filter(|a| cmd(a, "kind") == 2) {
        let s = cmd(a, "source") as usize; let t = cmd(a, "target") as usize;
        if s >= n || t >= n { continue; }
        let (su, tu) = match (&g.units[s], &g.units[t]) { (Some(x), Some(y)) => (x, y), _ => continue };
        let (aa, _, ahm, arange, _) = ustats(su.kind);
        let (_, dd, dhm, drange, dcost) = ustats(tu.kind);
        if aa <= 0.0 { continue; }
        let am = ahm + if su.veteran { 5.0 } else { 0.0 };
        let dm = dhm + if tu.veteran { 5.0 } else { 0.0 };
        let (dmg, ret) = combat(aa, su.hp, am, dd, tu.hp, dm, g.def_bonus(t, tu));
        let kill = tu.hp - dmg <= 0.0;
        let dist = g.d(s, t);
        let retal = if kill || stiff(tu.kind) || dist > drange { 0.0 } else { ret };
        let _ = arange;
        let mut score = if kill { 60.0 + dcost as f64 * 3.0 } else { dmg * 2.0 - retal * 1.2 };
        if !kill && su.hp - retal <= 0.0 { score -= 100.0; }
        if g.tiles[t].structure == 1 && g.tiles[t].owner != g.me && !kill { score += 5.0; }
        if score > 0.0 && best_attack.as_ref().map_or(true, |b| score > b.0) {
            best_attack = Some((score, id(a)));
        }
    }
    if let Some((_, a)) = best_attack { return a; }

    // 5. free monuments
    if let Some(a) = legal.iter().find(|a| cmd(a, "kind") == 6 && (23..=29).contains(&cmd(a, "parameter"))) { return id(a); }

    // 5b. thin army with enemies around: units first
    let summons: Vec<&Value> = legal.iter().filter(|a| cmd(a, "kind") == 4).collect();
    let count_kind = |k: i64| my_units.iter().filter(|&&i| g.units[i].as_ref().map_or(false, |u| u.kind == k)).count() as i64;
    let pick_summon = |summons: &Vec<&Value>| -> Option<String> {
        if summons.is_empty() { return None; }
        let avail = |k: i64| summons.iter().any(|a| cmd(a, "parameter") == k);
        let mut want: Vec<i64> = Vec::new();
        if avail(4) && nunits >= 3 { want.push(4); }
        if avail(12) { want.push(12); }
        if avail(11) { want.push(11); }
        if enemy_seen {
            if avail(5) && count_kind(5) < ncity + 1 { want.push(5); }
            if avail(9) && count_kind(9) * 3 < nunits { want.push(9); }
            if avail(2) { want.push(2); }
            if avail(5) { want.push(5); }
            if avail(9) { want.push(9); }
            if avail(3) { want.push(3); }
        } else {
            if avail(3) && count_kind(3) < 2 { want.push(3); }
            if avail(2) { want.push(2); }
            if avail(3) { want.push(3); }
            if avail(9) { want.push(9); }
            if avail(5) { want.push(5); }
        }
        for k in want {
            let mut best: Option<(i32, String)> = None;
            for a in summons.iter().filter(|a| cmd(a, "parameter") == k) {
                let s = cmd(a, "source") as usize;
                if s >= n { continue; }
                let mut sc = g.unknown_around(s) * 2;
                if let Some(&e) = enemy_units.iter().min_by_key(|&&e| g.d(e, s)) { sc += 20 - g.d(e, s); }
                if best.as_ref().map_or(true, |b| sc > b.0) { best = Some((sc, id(a))); }
            }
            if let Some((_, a)) = best { return Some(a); }
        }
        None
    };
    let min_army = if enemy_seen { ncity * 2 + 1 } else { ncity + 2 };
    if nunits < min_army {
        if let Some(a) = pick_summon(&summons) { return a; }
    }
    // 6. economy: harvest / buildings with best pop per star
    let has_port = |t: usize| (0..n).any(|i| g.tiles[i].structure == 8 && g.tiles[i].owner == g.me && g.d(i, t) <= 3);
    let adj_count = |t: usize, st: i64| g.neighbors(t).iter().filter(|&&j| g.tiles[j].structure == st && g.tiles[j].owner == g.me).count() as f64;
    let mut best_econ: Option<(f64, String)> = None;
    for a in legal.iter() {
        let k = cmd(a, "kind");
        let t = cmd(a, "target");
        if t < 0 || t as usize >= n { continue; }
        let t = t as usize;
        let ratio = if k == 5 { 0.5 } else if k == 6 {
            match cmd(a, "parameter") {
                12 => 1.0 / 3.0, 5 | 21 => 0.4,
                13 => { let c = adj_count(t, 12); if c >= 2.0 { c / 5.0 } else { 0.0 } }
                6 => { let c = adj_count(t, 5); if c >= 2.0 { c / 5.0 } else { 0.0 } }
                22 => { let c = adj_count(t, 21); if c >= 1.0 { 2.0 * c / 5.0 } else { 0.0 } }
                8 => if water_frac > 0.08 && !has_port(t) && g.stars >= 7 { 0.2 } else { 0.0 },
                17 | 18 | 19 | 20 => if g.stars >= 30 { 0.06 } else { 0.0 },
                50 => { let c = adj_count(t, 13) + adj_count(t, 6) + adj_count(t, 22); if c >= 2.0 { c / 6.0 } else { 0.0 } }
                _ => 0.0,
            }
        } else { 0.0 };
        if ratio > 0.0 && best_econ.as_ref().map_or(true, |b| ratio > b.0) { best_econ = Some((ratio, id(a))); }
    }
    if let Some((_, a)) = best_econ { return a; }


    let cap = ncity * 3 + 3 + if enemy_seen { ncity + 4 } else { 0 };
    // 8. research
    let mut prio: Vec<i64> = vec![1, 6, 16, 7, 8, 20, 18, 23, 24, 2, 3, 4, 17, 9, 21, 22, 5];
    if enemy_seen && !g.tech[7] { prio.retain(|&p| p != 7); prio.insert(0, 7); }
    if water_frac > 0.3 {
        prio.insert(0, 10);
        prio.insert(3, 13);
    } else if water_frac > 0.1 {
        prio.insert(1, 10);
        if ocean_seen { prio.insert(5, 13); }
    } else if water_frac > 0.04 {
        prio.insert(4, 10);
    }
    let researches: Vec<&Value> = legal.iter().filter(|a| cmd(a, "kind") == 7).collect();
    if !researches.is_empty() {
        for p in &prio {
            if let Some(a) = researches.iter().find(|a| cmd(a, "parameter") == *p) {
                let tier = match *p { 1 | 6 | 10 | 15 | 20 => 1, 2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2, _ => 3 };
                let cost = 4 + ncity * tier;
                // keep research flowing but not at the expense of a tiny army when enemies are near
                let allowed = g.stars >= cost && (ntech < 3 || summons.is_empty() || nunits >= cap - 1 || g.stars >= cost + 8);
                if allowed {
                    return id(a);
                }
                break;
            }
        }
    }

    // 9. summon up to cap

    if nunits < cap {
        if let Some(a) = pick_summon(&summons) { return a; }
    }
    // spend excess stars on research anyway
    if g.stars >= 15 {
        for p in &prio {
            if let Some(a) = researches.iter().find(|a| cmd(a, "parameter") == *p) { return id(a); }
        }
    }

    // 10. movement
    // goals: (tile, value)
    let mut goals: Vec<(usize, f64)> = Vec::new();
    for i in 0..n {
        let t = &g.tiles[i];
        if !t.known { continue; }
        if g.own_unit_at(i) { continue; }
        if t.structure == 1 && t.owner != g.me {
            if t.owner == 0 { goals.push((i, 50.0)); } else if aggressive { goals.push((i, 45.0)); }
        } else if t.structure == 2 {
            goals.push((i, 42.0));
        }
    }
    for &e in &enemy_units {
        if my_units.iter().any(|&m| g.d(m, e) <= 2) { goals.push((e, if aggressive { 20.0 } else { 10.0 })); }
    }
    // threatened own cities
    let mut threatened: Vec<usize> = Vec::new();
    for &(c, _) in &my_cities {
        if enemy_units.iter().any(|&e| g.d(e, c) <= 3) { threatened.push(c); }
    }
    for &c in &threatened { if !g.own_unit_at(c) { goals.push((c, 70.0)); } }
    if enemy_seen {
        for &(c, _) in &my_cities {
            if !threatened.contains(&c) && !g.own_unit_at(c) { goals.push((c, 38.0)); }
        }
    }
    // frontier attraction: distance to nearest unknown tile (land / naval)
    let frontier = |naval: bool| -> Vec<i32> {
        let mut dist = vec![INF; n];
        let mut q = std::collections::VecDeque::new();
        for i in 0..n { if !g.tiles[i].known { dist[i] = 0; q.push_back(i); } }
        while let Some(x) = q.pop_front() {
            if g.tiles[x].known {
                let ok = if naval { g.water_pass(x) && !g.is_land(x) } else { g.land_pass(x) };
                if !ok { continue; }
            }
            for y in g.neighbors(x) {
                if dist[y] != INF { continue; }
                let enter = if naval { g.water_pass(y) || (g.is_land(y) && !g.enemy_unit_at(y)) } else { g.land_pass(y) };
                if enter { dist[y] = dist[x] + 1; q.push_back(y); }
            }
        }
        dist
    };
    let front_land = frontier(false);
    let front_naval = frontier(true);

    let land_maps: Vec<(usize, f64, Vec<i32>)> = goals.iter().map(|&(t, v)| (t, v, g.bfs(t, false))).collect();
    let naval_maps: Vec<(usize, f64, Vec<i32>)> = goals.iter().map(|&(t, v)| (t, v, g.bfs(t, true))).collect();
    // own ports as embark goals for land units
    let ports: Vec<usize> = (0..n).filter(|&i| g.tiles[i].structure == 8 && g.tiles[i].owner == g.me).collect();
    let port_maps: Vec<Vec<i32>> = ports.iter().map(|&p| g.bfs(p, false)).collect();

    let value_at = |t: usize, naval: bool, unit_hp_frac: f64| -> f64 {
        let maps = if naval { &naval_maps } else { &land_maps };
        let mut best = -1e9f64;
        let mut any_reach = false;
        for (_, v, m) in maps.iter() {
            if m[t] < INF {
                any_reach = true;
                let val = v - 5.0 * m[t] as f64;
                if val > best { best = val; }
            }
        }
        if !naval && !any_reach {
            for pm in port_maps.iter() {
                if pm[t] < INF { let val = 20.0 - 4.0 * pm[t] as f64; if val > best { best = val; } }
            }
        }
        let fd = if naval { front_naval[t] } else { front_land[t] };
        if fd < INF { let val = (if enemy_seen && !aggressive { 8.0 } else { 18.0 }) - 3.0 * fd as f64; if val > best { best = val; } }
        if best < -1e8 { best = 0.0; }
        if enemy_seen && !aggressive {
            let dc = my_cities.iter().map(|&(c, _)| g.d(c, t)).min().unwrap_or(0);
            if dc > 3 { best -= 6.0 * (dc - 3) as f64; }
        }
        // danger: adjacent enemies when weak
        if unit_hp_frac < 0.5 {
            let adj_en = enemy_units.iter().filter(|&&e| g.d(e, t) <= 1).count() as f64;
            best -= adj_en * 8.0;
        }
        best
    };

    let mut best_move: Option<(f64, String)> = None;
    let need_room = summons.is_empty() && g.stars >= 3 && nunits < cap;
    let mut cur_cache: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for a in legal.iter().filter(|a| cmd(a, "kind") == 1) {
        let s = cmd(a, "source") as usize; let t = cmd(a, "target") as usize;
        if s >= n || t >= n { continue; }
        let u = match &g.units[s] { Some(u) => u, None => continue };
        let naval = is_naval(u.kind);
        let (_, _, hm, _, _) = ustats(u.kind);
        let frac = u.hp / (hm + if u.veteran { 5.0 } else { 0.0 });
        // stay put in a threatened city
        if threatened.contains(&s) { continue; }
        if u.kind == 5 && !(g.tiles[t].structure == 1 && g.tiles[t].owner == g.me) { continue; }
        // make room for a new recruit: step off the city tile when nothing can be summoned
        let on_city = g.tiles[s].structure == 1 && g.tiles[s].owner == g.me;
        if on_city && need_room && !enemy_units.iter().any(|&e| g.d(e, s) <= 2) {
            if g.d(s, t) != 1 { continue; }
            let adj_en = enemy_units.iter().filter(|&&e| g.d(e, t) <= 1).count() as f64;
            let sc = 12.0 - adj_en * 4.0 + g.unknown_around(t) as f64;
            if best_move.as_ref().map_or(true, |b| sc > b.0) { best_move = Some((sc, id(a))); }
            continue;
        }
        // garrison: keep a unit in a city when enemies are around (within 6) and no other own unit is inside/adjacent
        if on_city && enemy_seen {
            let near_enemy = enemy_units.iter().any(|&e| g.d(e, s) <= 6);
            let other_guard = g.neighbors(s).iter().any(|&j| g.own_unit_at(j));
            if near_enemy && (!other_guard || u.kind == 5) { continue; }
            if u.kind == 5 { continue; }
        }
        let cur = *cur_cache.entry(s).or_insert_with(|| value_at(s, naval, frac));
        let mut v = value_at(t, naval, frac) + g.unknown_around(t) as f64 * 3.0;
        // small preference for defensive terrain / staying off water for land units
        if g.tiles[t].terrain == 5 || g.tiles[t].terrain == 4 { v += 0.5; }
        let gain = v - cur;
        if gain > 0.5 && best_move.as_ref().map_or(true, |b| gain > b.0) { best_move = Some((gain, id(a))); }
    }
    if let Some((_, a)) = best_move { return a; }

    // 11. end turn
    if let Some(a) = legal.iter().find(|a| cmd(a, "kind") == 10) { return id(a); }
    id(&legal[0])
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let a = choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": a})).unwrap();
        stdout.flush().unwrap();
    }
}
