use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

struct Tile { known: bool, terrain: i32, owner: i32, structure: i32, resource: i32 }
struct Unit { tile: usize, owner: i32, kind: i32, health: f64, veteran: bool, moved: bool, attacked: bool }
struct City { tile: usize, owner: i32 }
struct Act { id: String, kind: i64, source: i64, target: i64, param: i64 }

struct Bot { declined: HashSet<usize>, declined_turn: i64, enemy_cities: HashSet<usize>, contact: bool, research_turn: i64 }

// (attack, defense, hp, range, stiff, cost)
fn stats(kind: i32) -> (f64, f64, f64, i32, bool, i32) {
    match kind {
        2 => (2.0, 2.0, 10.0, 1, false, 2),
        3 => (2.0, 1.0, 10.0, 1, false, 3),
        4 => (3.5, 1.0, 10.0, 1, false, 8),
        5 => (1.0, 3.0, 15.0, 1, false, 3),
        8 => (4.0, 0.0, 10.0, 3, true, 8),
        9 => (2.0, 1.0, 10.0, 2, false, 3),
        10 => (0.0, 1.0, 10.0, 1, true, 5),
        11 => (3.0, 3.0, 15.0, 1, false, 5),
        12 => (5.0, 4.0, 40.0, 1, false, 10),
        37 => (2.0, 2.0, 10.0, 1, false, 2),
        38 => (0.0, 0.5, 5.0, 1, true, 8),
        39 => (0.0, 0.5, 5.0, 1, true, 2),
        40 => (2.0, 2.0, 10.0, 1, false, 2),
        41 => (3.0, 2.0, 10.0, 3, true, 15),
        42 => (2.0, 1.0, 10.0, 2, false, 5),
        43 => (0.0, 1.0, 10.0, 2, true, 0),
        44 => (3.0, 3.0, 10.0, 1, false, 5),
        45 => (4.0, 4.0, 40.0, 1, true, 10),
        _ => (2.0, 2.0, 10.0, 1, false, 2),
    }
}
fn is_naval(kind: i32) -> bool { (41..=45).contains(&kind) || kind == 39 || kind == 40 }

fn combat(aa: f64, ah: f64, am: f64, dd: f64, dh: f64, dm: f64, db: f64) -> (f64, f64) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    if total <= 0.0 { return (0.0, 0.0); }
    let attack = ((af / total) * aa * 4.5 + 0.000001 + 0.5).floor();
    let defense = if dh > attack { ((df / total) * dd * 4.5 + 0.000001 + 0.5).floor() } else { 0.0 };
    (attack, defense)
}

fn cheb(a: usize, b: usize, size: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

fn neighbors(t: usize, size: usize) -> Vec<usize> {
    let (x, y) = ((t % size) as i32, (t / size) as i32);
    let mut v = Vec::with_capacity(8);
    for dy in -1..=1 { for dx in -1..=1 {
        if dx == 0 && dy == 0 { continue; }
        let (nx, ny) = (x + dx, y + dy);
        if nx >= 0 && ny >= 0 && nx < size as i32 && ny < size as i32 { v.push((ny as usize) * size + nx as usize); }
    }}
    v
}

// Multi-source BFS over passable tiles.
fn bfs(sources: &[usize], passable: &[bool], size: usize) -> Vec<i32> {
    let n = size * size;
    let mut d = vec![INF; n];
    let mut q = std::collections::VecDeque::new();
    for &s in sources { if d[s] == INF { d[s] = 0; q.push_back(s); } }
    while let Some(t) = q.pop_front() {
        for nb in neighbors(t, size) {
            if d[nb] == INF && passable[nb] { d[nb] = d[t] + 1; q.push_back(nb); }
        }
    }
    d
}

impl Bot {
    fn choose(&mut self, obs: &Value) -> String {
        let legal_v = obs["legal_actions"].as_array().cloned().unwrap_or_default();
        let acts: Vec<Act> = legal_v.iter().map(|a| Act {
            id: a["action_id"].as_str().unwrap_or("").to_string(),
            kind: a["command"]["kind"].as_i64().unwrap_or(0),
            source: a["command"]["source"].as_i64().unwrap_or(-1),
            target: a["command"]["target"].as_i64().unwrap_or(-1),
            param: a["command"]["parameter"].as_i64().unwrap_or(0),
        }).collect();
        if acts.is_empty() { return String::new(); }
        let fallback = acts.iter().find(|a| a.kind == 10).unwrap_or(&acts[0]).id.clone();
        let me = obs["player"].as_i64().unwrap_or(1) as i32;
        let vs = &obs["visible_state"];
        let size = vs["board_size"].as_u64().unwrap_or(14) as usize;
        let n = size * size;
        let turn = vs["turn"].as_i64().unwrap_or(0);
        let stars = vs["stars"].as_i64().unwrap_or(0);
        let techs: HashSet<i64> = vs["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
        let mut tiles: Vec<Tile> = (0..n).map(|_| Tile { known: false, terrain: 0, owner: 0, structure: 0, resource: 0 }).collect();
        for t in vs["tiles"].as_array().unwrap_or(&vec![]) {
            let i = t["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n { continue; }
            let known = t["known"].as_bool().unwrap_or(false);
            tiles[i] = Tile { known, terrain: t["terrain"].as_i64().unwrap_or(0) as i32, owner: t["owner"].as_i64().unwrap_or(0) as i32,
                structure: t["structure"].as_i64().unwrap_or(0) as i32, resource: t["resource"].as_i64().unwrap_or(0) as i32 };
        }
        let units: Vec<Unit> = vs["units"].as_array().unwrap_or(&vec![]).iter().map(|u| Unit {
            tile: u["tile"].as_u64().unwrap_or(0) as usize, owner: u["owner"].as_i64().unwrap_or(0) as i32,
            kind: u["kind"].as_i64().unwrap_or(2) as i32, health: u["health"].as_f64().unwrap_or(10.0),
            veteran: u["veteran"].as_bool().unwrap_or(false), moved: u["moved"].as_bool().unwrap_or(false),
            attacked: u["attacked"].as_bool().unwrap_or(false) }).collect();
        let cities: Vec<City> = vs["cities"].as_array().unwrap_or(&vec![]).iter().map(|c| City {
            tile: c["tile"].as_u64().unwrap_or(0) as usize, owner: c["owner"].as_i64().unwrap_or(0) as i32 }).collect();
        let my_cities: Vec<usize> = cities.iter().filter(|c| c.owner == me).map(|c| c.tile).collect();
        let n_cities = my_cities.len().max(1) as i64;
        let my_units: Vec<&Unit> = units.iter().filter(|u| u.owner == me).collect();
        let enemy_units: Vec<&Unit> = units.iter().filter(|u| u.owner != me).collect();
        let unit_at = |t: usize| units.iter().find(|u| u.tile == t);
        if turn != self.declined_turn { self.declined.clear(); self.declined_turn = turn; }
        for c in &cities { if c.owner != me && c.owner != 0 { self.enemy_cities.insert(c.tile); } }
        for i in 0..n { if tiles[i].known && tiles[i].structure == 1 && tiles[i].owner == me { self.enemy_cities.remove(&i); } }
        if !enemy_units.is_empty() || !self.enemy_cities.is_empty() { self.contact = true; }
        let known_count = tiles.iter().filter(|t| t.known).count().max(1);
        let water_count = tiles.iter().filter(|t| t.known && (t.terrain == 1 || t.terrain == 2)).count();
        let water_frac = water_count as f64 / known_count as f64;
        let mountain_count = tiles.iter().filter(|t| t.known && t.terrain == 4).count();
        let my_ports = (0..n).filter(|&i| tiles[i].structure == 8 && tiles[i].owner == me).count();

        // 1. City rewards
        if acts.iter().any(|a| a.kind == 9) {
            let prefs: [i64; 8] = if turn < 8 { [6, 8, 1, 4, 3, 7, 5, 2] } else { [6, 8, 1, 3, 7, 5, 4, 2] };
            for pref in prefs {
                if let Some(a) = acts.iter().find(|a| a.kind == 9 && a.param == pref) { return a.id.clone(); }
            }
        }
        // 2. Capture
        if let Some(a) = acts.iter().find(|a| a.kind == 8) { return a.id.clone(); }
        // 3. Promote
        if let Some(a) = acts.iter().find(|a| a.kind == 3 && a.param == 16) { return a.id.clone(); }

        // 4. Attacks
        let mut best_attack: Option<(f64, &Act)> = None;
        for a in acts.iter().filter(|a| a.kind == 2) {
            let (s, t) = (a.source as usize, a.target as usize);
            let (Some(u), Some(e)) = (unit_at(s), unit_at(t)) else { continue; };
            let (aa, _, ahm, arange, _, _) = stats(u.kind);
            let (ea, ed, ehm, erange, estiff, ecost) = stats(e.kind);
            let am = ahm + if u.veteran { 5.0 } else { 0.0 };
            let dm = ehm + if e.veteran { 5.0 } else { 0.0 };
            let tt = &tiles[t];
            let db = if tt.terrain == 5 || tt.terrain == 4 { 1.4 } else if tt.structure == 1 && tt.owner == e.owner { 1.5 } else { 1.0 };
            let (dmg, ret) = combat(aa, u.health, am, ed, e.health, dm, db);
            let dist = cheb(s, t, size);
            let kill = dmg >= e.health;
            let retal = if kill || estiff || dist > erange || ea <= 0.0 { 0.0 } else { ret };
            let mut score = if kill { 60.0 + ecost as f64 * 4.0 } else { dmg * 2.5 - retal * 1.5 };
            if u.health - retal <= 0.0 { score -= 500.0; }
            if arange >= 2 && !kill { score += 5.0; }
            // enemy sitting on my city: attack strongly
            if tt.structure == 1 && tt.owner == me { score += 20.0; }
            if tiles[s].structure == 1 && tiles[s].owner == me && !kill { score -= 2.0; }
            if tiles[s].structure == 1 && tiles[s].owner == me && kill && arange == 1 && (u.kind == 5 || enemy_units.len() > 2) { score -= 30.0; }
            if best_attack.map_or(true, |(bs, _)| score > bs) { best_attack = Some((score, a)); }
        }
        if let Some((s, a)) = best_attack { if s > 0.0 { return a.id.clone(); } }

        // threat assessment
        let threatened_city = my_cities.iter().any(|&c| enemy_units.iter().any(|e| cheb(e.tile, c, size) <= 2) && unit_at(c).map_or(true, |u| u.owner != me));
        let army = my_units.iter().filter(|u| !is_naval(u.kind) || u.kind == 43).count() as i64;
        let cap = if self.contact { n_cities * 4 + 4 } else { n_cities * 2 + 2 };

        // role: 0 = expand (pre-contact), 1 = defend, 2 = attack
        let recruit = |min_stars: i64, role: i32| -> Option<String> {
            let rec: Vec<&Act> = acts.iter().filter(|a| a.kind == 4 && a.param != 43 && a.param != 44 && a.param != 42 && a.param != 41).collect();
            if rec.is_empty() { return None; }
            let mut best_city: Option<(i32, usize)> = None;
            for a in &rec {
                let c = a.source as usize;
                let mut d = 50;
                for e in &enemy_units { d = d.min(cheb(e.tile, c, size)); }
                for &ec in &self.enemy_cities { d = d.min(cheb(ec, c, size) + 3); }
                if best_city.map_or(true, |(bd, _)| d < bd) { best_city = Some((d, c)); }
            }
            let city = best_city?.1;
            let has = |k: i64| rec.iter().find(|a| a.source as usize == city && a.param == k).map(|a| a.id.clone());
            let s = stars - min_stars;
            let order: Vec<(i64, i64)> = match role {
                0 => vec![(3, 3), (2, 2)],
                1 => vec![(5, 3), (11, 5), (2, 2), (9, 3), (3, 3)],
                _ => {
                    let strong = techs.contains(&3) || techs.contains(&24);
                    if strong && !(threatened_city && stars < 5) { vec![(4, 8), (11, 5)] } else { vec![(4, 8), (11, 5), (9, 3), (3, 3), (2, 2)] }
                }
            };
            let archers = my_units.iter().filter(|u| u.kind == 9).count();
            for (k, cost) in order {
                if s < cost { continue; }
                if k == 9 && archers * 2 > my_units.len() { continue; }
                if let Some(id) = has(k) { return Some(id); }
            }
            None
        };
        let ungarrisoned = my_cities.iter().any(|&c| unit_at(c).is_none());
        let n_def0 = my_units.iter().filter(|u| u.kind == 5).count() as i64;
        if threatened_city { if let Some(id) = recruit(0, if n_def0 < n_cities { 1 } else { 2 }) { return id; } }
        let n_def = my_units.iter().filter(|u| u.kind == 5).count() as i64;
        let need_def = ungarrisoned && n_def < n_cities;
        if self.contact && need_def && stars >= 3 { if let Some(id) = recruit(0, 1) { return id; } }

        // 5. Research (early / cheap)
        let has_fish = techs.contains(&10);
        let has_climb = techs.contains(&20);
        let land_pass: Vec<bool> = (0..n).map(|i| {
            let t = &tiles[i];
            if !t.known { return true; }
            match t.terrain { 3 | 5 => true, 4 => has_climb, 1 | 2 => (has_fish && my_ports > 0) || t.structure == 48 || (t.structure == 8 && t.owner == me), _ => false }
        }).collect();
        let stuck = {
            let pure_land: Vec<bool> = (0..n).map(|i| { let t = &tiles[i]; !t.known || t.terrain == 3 || t.terrain == 5 || (t.terrain == 4 && has_climb) || t.structure == 48 }).collect();
            let d = bfs(&my_cities, &pure_land, size);
            let unknown_reach = (0..n).any(|i| !tiles[i].known && d[i] < INF);
            let enemy_reach = self.enemy_cities.iter().any(|&c| d[c] < INF) || enemy_units.iter().any(|e| d[e.tile] < INF);
            !unknown_reach && !enemy_reach && water_count > 0
        };
        let owned = |f: &dyn Fn(&Tile) -> bool| tiles.iter().filter(|t| t.known && t.owner == me && f(t)).count() as f64;
        let forests = owned(&|t| t.terrain == 5 && t.structure == 0);
        let fruits = owned(&|t| t.resource == 6);
        let crops = owned(&|t| t.resource == 2);
        let fish = owned(&|t| t.resource == 3);
        let metal = owned(&|t| t.resource == 5);
        let mountains = owned(&|t| t.terrain == 4);
        let waters = owned(&|t| t.terrain == 1);
        let huts = owned(&|t| t.structure == 12);
        let farms = owned(&|t| t.structure == 5);
        let c = self.contact;
        let mut scored: Vec<(f64, i64)> = vec![
            (if c { 1.0 } else { 3.0 }, 1),
            (0.9 + fruits * 1.0 + if c { 1.0 } else { 0.0 }, 6),
            (0.5 + forests * 0.8, 16),
            (if c { 2.5 } else { 0.8 }, 7),
            (0.3 + crops * 1.5 + 0.3 * forests, 8),
            (0.2 + fish * 1.0 + waters * 0.4 + if water_frac >= 0.25 || stuck { 3.0 } else { 0.0 }, 10),
            (0.2 + mountains * 0.4 + metal * 0.5, 20),
            (0.2 + metal * 1.5, 23),
            (if c { 2.0 } else { 0.5 }, 24),
            (if c { 2.2 } else { 0.3 }, 2),
            (if c { 3.0 } else { 0.8 }, 3),
            (if c { 1.0 } else { 0.4 }, 18),
            (0.2 + if huts >= 2.0 { huts * 0.8 } else { 0.0 }, 17),
            (0.2 + if farms >= 2.0 { farms * 0.8 } else { 0.0 }, 9),
            (if water_frac >= 0.25 { 1.5 } else { 0.1 }, 13),
            (0.1, 21), (0.1, 22), (0.05, 4), (0.05, 5), (0.1, 19), (0.1, 44), (0.1, 14), (0.1, 12),
        ];
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let prio: Vec<i64> = scored.iter().map(|x| x.1).collect();
        let tier_of = |p: i64| -> i64 { match p { 1 | 6 | 10 | 15 | 20 => 1, 2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2, _ => 3 } };
        let requires = |p: i64| -> i64 { match p { 2 | 4 => 1, 7 | 8 => 6, 13 | 44 => 10, 16 | 18 => 15, 21 | 23 => 20, 3 => 2, 5 => 4, 9 => 8, 17 => 16, 19 => 18, 22 => 21, 24 => 23, 12 | 14 => 44, 38 => 7, _ => -1 } };
        let research_pick = |reserve: i64| -> Option<String> {
            if self.research_turn == turn { return None; }
            for &p in &prio {
                if techs.contains(&p) { continue; }
                let req = requires(p);
                if req >= 0 && !techs.contains(&req) { continue; }
                let cost = 4 + n_cities * tier_of(p);
                if let Some(a) = acts.iter().find(|a| a.kind == 7 && a.param == p) {
                    if stars - cost >= reserve { return Some(a.id.clone()); }
                }
                return None;
            }
            None
        };
        if stars >= 10 && !threatened_city { if let Some(id) = research_pick(4) { self.research_turn = turn; return id; } }
        if !techs.contains(&1) && turn <= 4 && !self.contact { if let Some(a) = acts.iter().find(|a| a.kind == 7 && a.param == 1) { self.research_turn = turn; return a.id.clone(); } }

        // 6. Economy: harvest / build
        let mut best_econ: Option<(f64, &Act)> = None;
        for a in &acts {
            let v = match a.kind {
                5 => 0.5,
                6 => {
                    let t = a.target as usize;
                    let adj_count = |kinds: &[i32]| neighbors(t, size).iter().filter(|&&j| tiles[j].owner == me && kinds.contains(&tiles[j].structure)).count() as f64;
                    match a.param {
                        12 => 1.0 / 3.0,
                        5 | 21 => 0.4,
                        8 => if (water_frac >= 0.2 && my_ports == 0) || (stuck && (my_ports as i64) < n_cities) || (water_frac >= 0.35 && (my_ports as i64) < n_cities) { 0.45 } else { 0.12 },
                        13 => { let c = adj_count(&[12]); if c >= 2.0 { c / 5.0 } else { 0.0 } }
                        6 => { let c = adj_count(&[5]); if c >= 2.0 { c / 5.0 } else { 0.0 } }
                        22 => { let c = adj_count(&[21]); if c >= 1.0 { 2.0 * c / 5.0 } else { 0.0 } }
                        17 | 18 | 19 | 20 => if stars >= 22 { 0.3 } else { 0.0 },
                        _ => 0.0,
                    }
                }
                _ => 0.0,
            };
            if v > 0.0 && best_econ.map_or(true, |(bv, _)| v > bv) { best_econ = Some((v, a)); }
        }
        if let Some((v, a)) = best_econ { if v >= 0.3 { return a.id.clone(); } }

        // 10. Movement
        let naval_pass: Vec<bool> = (0..n).map(|i| { let t = &tiles[i]; !t.known || t.terrain == 1 || t.terrain == 2 || t.terrain == 3 || t.terrain == 5 || (t.terrain == 4 && has_climb) }).collect();
        // objectives
        let mut objectives: Vec<(usize, i32)> = Vec::new(); // (tile, weight)
        for i in 0..n {
            let t = &tiles[i];
            if !t.known { continue; }
            if t.structure == 2 { objectives.push((i, 0)); }
            if t.structure == 1 && t.owner != me {
                let enemy_here = unit_at(i).map_or(false, |u| u.owner != me);
                objectives.push((i, if t.owner == 0 { 0 } else if enemy_here { 3 } else { 1 }));
            }
        }
        for e in &enemy_units { if tiles[e.tile].structure != 1 { objectives.push((e.tile, 2)); } }
        let unknown: Vec<usize> = (0..n).filter(|&i| !tiles[i].known).collect();
        let can_recruit_more = stars >= 5 && army < cap;
        let garrison = |u: &Unit| -> bool {
            let on_city = my_cities.contains(&u.tile);
            if !on_city { return false; }
            if can_recruit_more { return false; }
            u.kind == 5 || u.kind == 12 || enemy_units.iter().any(|e| cheb(e.tile, u.tile, size) <= 3)
        };
        let movers: Vec<&Unit> = my_units.iter().cloned().filter(|u| !u.moved && !self.declined.contains(&u.tile) && !garrison(u) && acts.iter().any(|a| a.kind == 1 && a.source as usize == u.tile)).collect();
        // defenders: hold cities; shuffle off to let the city recruit
        if let Some(u) = movers.iter().find(|u| u.kind == 5 || (u.kind == 12 && !self.contact)) {
            let on_city = my_cities.contains(&u.tile);
            let empty_cities: Vec<usize> = my_cities.iter().cloned().filter(|&c| unit_at(c).is_none()).collect();
            let mut best: Option<(i32, &Act)> = None;
            for a in acts.iter().filter(|a| a.kind == 1 && a.source as usize == u.tile) {
                let t = a.target as usize;
                let tt = &tiles[t];
                let mut sc = 0;
                if !empty_cities.is_empty() {
                    let dn = empty_cities.iter().map(|&c| cheb(c, t, size)).min().unwrap();
                    sc -= dn * 10;
                    if empty_cities.contains(&t) { sc += 50; }
                } else if on_city {
                    // step just off the city, prefer defensive terrain, stay adjacent
                    let dc = cheb(u.tile, t, size);
                    sc -= dc * 10;
                    if tt.terrain == 5 || tt.terrain == 4 { sc += 3; }
                    if tt.owner != me { sc -= 5; }
                } else {
                    // return towards nearest own city
                    let dn = my_cities.iter().map(|&c| cheb(c, t, size)).min().unwrap_or(0);
                    sc -= dn * 10;
                }
                if best.map_or(true, |(bs, _)| sc > bs) { best = Some((sc, a)); }
            }
            let cur = if !empty_cities.is_empty() { -(empty_cities.iter().map(|&c| cheb(c, u.tile, size)).min().unwrap()) * 10 + if empty_cities.contains(&u.tile) { 50 } else { 0 } }
                      else if on_city { -1000 } else { -(my_cities.iter().map(|&c| cheb(c, u.tile, size)).min().unwrap_or(0)) * 10 };
            if let Some((sc, a)) = best { if sc > cur { return a.id.clone(); } }
            self.declined.insert(u.tile);
            return self.choose(obs);
        }
        if let Some(u) = movers.first() {
            let pass = if is_naval(u.kind) { &naval_pass } else { &land_pass };
            let mut pass_u = pass.clone();
            pass_u[u.tile] = true;
            // recover if badly hurt and safe
            let (_, _, hm, _, _, _) = stats(u.kind);
            let maxhp = hm + if u.veteran { 5.0 } else { 0.0 };
            let near_enemy = enemy_units.iter().any(|e| cheb(e.tile, u.tile, size) <= 2);
            if u.health < 0.55 * maxhp && !near_enemy {
                if let Some(a) = acts.iter().find(|a| a.kind == 3 && a.param == 7 && a.source as usize == u.tile) { return a.id.clone(); }
            }
            // choose goal: nearest objective not claimed by a closer idle unit
            let mut best_goal: Option<(i32, Vec<i32>)> = None;
            for &(obj, w) in &objectives {
                let d = bfs(&[obj], &pass_u, size);
                let du = d[u.tile];
                if du >= INF { continue; }
                // claimed if another of my units is strictly closer (by chebyshev) and not naval raft
                let claimed = my_units.iter().any(|o| o.tile != u.tile && cheb(o.tile, obj, size) < du && w <= 1 && o.kind != 43);
                if claimed { continue; }
                let score = du + w * 2;
                if best_goal.as_ref().map_or(true, |(bs, _)| score < *bs) { best_goal = Some((score, d)); }
            }
            let dist_map: Option<Vec<i32>> = match best_goal {
                Some((_, d)) => Some(d),
                None => {
                    if !unknown.is_empty() {
                        let d = bfs(&unknown, &pass_u, size);
                        if d[u.tile] < INF { Some(d) } else { None }
                    } else { None }
                }
            };
            let dist_map = dist_map.or_else(|| {
                // fall back: head to remembered enemy cities (even through unknown pass)
                let ecs: Vec<usize> = self.enemy_cities.iter().cloned().collect();
                if ecs.is_empty() { None } else { let d = bfs(&ecs, &pass_u, size); if d[u.tile] < INF { Some(d) } else { None } }
            });
            if let Some(d) = dist_map {
                let cur = d[u.tile];
                let mut best: Option<(i32, &Act)> = None;
                for a in acts.iter().filter(|a| a.kind == 1 && a.source as usize == u.tile) {
                    let t = a.target as usize;
                    if d[t] >= INF { continue; }
                    let mut sc = -d[t] * 10;
                    let tt = &tiles[t];
                    if tt.structure == 1 && tt.owner != me { sc += 15; }
                    if tt.structure == 2 { sc += 15; }
                    if tt.terrain == 5 || tt.terrain == 4 { sc += 1; }
                    // avoid stacking adjacent to many enemies when weak
                    let adj_en = enemy_units.iter().filter(|e| cheb(e.tile, t, size) <= 1).count() as i32;
                    if u.health < 0.5 * maxhp { sc -= adj_en * 6; }
                    if best.map_or(true, |(bs, _)| sc > bs) { best = Some((sc, a)); }
                }
                if let Some((sc, a)) = best { if sc > -cur * 10 { return a.id.clone(); } }
            }
            self.declined.insert(u.tile);
            // continue to next mover on the next observation by recursing
            return self.choose(obs);
        }
        // 7. Recruit if army small
        let role = if !self.contact { 0 } else if need_def || (threatened_city && n_def < n_cities) { 1 } else { 2 };
        if army < cap { if let Some(id) = recruit(0, role) { return id; } }
        // 8. Research with remaining stars
        let res_reserve = if self.contact && stars < 15 { 3 } else { 0 };
        if let Some(id) = research_pick(res_reserve) { self.research_turn = turn; return id; }
        // 9. Temples / leftover econ when rich
        if let Some((v, a)) = best_econ { if v > 0.0 && stars >= 20 { return a.id.clone(); } }
        if stars >= 15 { if let Some(id) = recruit(8, role) { return id; } }

        fallback
    }
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut bot = Bot { declined: HashSet::new(), declined_turn: -1, enemy_cities: HashSet::new(), contact: false, research_turn: -1 };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = bot.choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
