use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

fn stats(kind: i64) -> (f64, f64, f64, i64, f64) {
    // attack, defense, maxhp, range, cost
    match kind {
        2 => (2.0, 2.0, 10.0, 1, 2.0),
        3 => (2.0, 1.0, 10.0, 1, 3.0),
        4 => (3.5, 1.0, 10.0, 1, 8.0),
        5 => (1.0, 3.0, 15.0, 1, 3.0),
        8 => (4.0, 0.0, 10.0, 3, 8.0),
        9 => (2.0, 1.0, 10.0, 2, 3.0),
        10 => (0.0, 1.0, 10.0, 1, 5.0),
        11 => (3.0, 3.0, 15.0, 1, 5.0),
        12 => (5.0, 4.0, 40.0, 1, 10.0),
        15 => (3.0, 1.0, 15.0, 1, 2.0),
        39 => (0.0, 0.5, 5.0, 1, 2.0),
        40 => (2.0, 2.0, 10.0, 1, 2.0),
        41 => (3.0, 2.0, 10.0, 3, 15.0),
        42 => (2.0, 1.0, 10.0, 2, 5.0),
        43 => (0.0, 1.0, 10.0, 2, 0.0),
        44 => (3.0, 3.0, 10.0, 1, 5.0),
        45 => (4.0, 4.0, 40.0, 1, 10.0),
        _ => (2.0, 2.0, 10.0, 1, 3.0),
    }
}
fn naval(kind: i64) -> bool { (39..=45).contains(&kind) }

#[derive(Clone, Default)]
struct Tile { known: bool, terrain: i64, owner: i64, structure: i64, resource: i64, capital: i64 }
#[derive(Clone)]
struct Unit { tile: i64, owner: i64, kind: i64, hp: f64, vet: bool, moved: bool, attacked: bool }

struct Bot { turn_seen: i64, }

fn cheb(a: i64, b: i64, n: i64) -> i64 { ((a % n) - (b % n)).abs().max((a / n - b / n).abs()) }

fn neighbors(t: i64, n: i64) -> Vec<i64> {
    let (x, y) = (t % n, t / n);
    let mut v = Vec::with_capacity(8);
    for dy in -1..=1 { for dx in -1..=1 {
        if dx == 0 && dy == 0 { continue; }
        let (nx, ny) = (x + dx, y + dy);
        if nx >= 0 && ny >= 0 && nx < n && ny < n { v.push(ny * n + nx); }
    }}
    v
}

// multi-source BFS with initial costs; passable(t)
fn field(n: i64, sources: &[(i64, i64)], pass: &dyn Fn(i64) -> bool) -> Vec<i64> {
    let nn = (n * n) as usize;
    let mut d = vec![i64::MAX / 4; nn];
    // Dial-ish: sort sources by cost, use a deque with simple relaxation (costs small)
    let mut q: VecDeque<i64> = VecDeque::new();
    let mut srcs = sources.to_vec();
    srcs.sort_by_key(|s| s.1);
    for &(t, c) in &srcs { if c < d[t as usize] { d[t as usize] = c; q.push_back(t); } }
    // Bellman-ford-like SPFA (unit edges)
    while let Some(t) = q.pop_front() {
        let dt = d[t as usize];
        for u in neighbors(t, n) {
            if !pass(u) { continue; }
            if dt + 1 < d[u as usize] { d[u as usize] = dt + 1; q.push_back(u); }
        }
    }
    d
}

impl Bot {
    fn choose(&mut self, obs: &Value) -> String {
        let legal = match obs["legal_actions"].as_array() { Some(l) if !l.is_empty() => l, _ => return String::new() };
        let me = obs["player"].as_i64().unwrap_or(1);
        let v = &obs["visible_state"];
        let n = v["board_size"].as_i64().unwrap_or(14);
        let nn = (n * n) as usize;
        let stars = v["stars"].as_i64().unwrap_or(0);
        let turn = v["turn"].as_i64().unwrap_or(0);
        self.turn_seen = turn;
        let techs: Vec<i64> = v["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
        let has = |t: i64| techs.contains(&t);
        let mut tiles = vec![Tile::default(); nn];
        if let Some(arr) = v["tiles"].as_array() {
            for t in arr {
                let i = t["tile"].as_i64().unwrap_or(-1);
                if i < 0 || i as usize >= nn { continue; }
                let g = |k: &str| t[k].as_i64().unwrap_or(0);
                tiles[i as usize] = Tile { known: t["known"].as_bool().unwrap_or(false), terrain: g("terrain"), owner: g("owner"),
                    structure: g("structure"), resource: g("resource"), capital: g("capital") };
            }
        }
        let mut units: Vec<Unit> = Vec::new();
        if let Some(arr) = v["units"].as_array() {
            for u in arr {
                units.push(Unit { tile: u["tile"].as_i64().unwrap_or(-1), owner: u["owner"].as_i64().unwrap_or(0), kind: u["kind"].as_i64().unwrap_or(2),
                    hp: u["health"].as_f64().unwrap_or(10.0), vet: u["veteran"].as_bool().unwrap_or(false),
                    moved: u["moved"].as_bool().unwrap_or(false), attacked: u["attacked"].as_bool().unwrap_or(false) });
            }
        }
        units.retain(|u| u.tile >= 0 && (u.tile as usize) < nn);
        let mut unit_at: Vec<Option<usize>> = vec![None; nn];
        for (i, u) in units.iter().enumerate() { if u.tile >= 0 && (u.tile as usize) < nn { unit_at[u.tile as usize] = Some(i); } }
        let enemies: Vec<&Unit> = units.iter().filter(|u| u.owner != me && u.owner > 0).collect();
        let mine_n = units.iter().filter(|u| u.owner == me).count() as i64;
        let my_cities: Vec<i64> = (0..nn as i64).filter(|&i| tiles[i as usize].structure == 1 && tiles[i as usize].owner == me).collect();
        let ncity = my_cities.len() as i64;
        let is_water = |t: i64| { let tr = tiles[t as usize].terrain; tiles[t as usize].known && (tr == 1 || tr == 2) };

        // city threat
        let mut threatened: Vec<i64> = Vec::new();
        for &c in &my_cities {
            if enemies.iter().any(|e| cheb(e.tile, c, n) <= 3) { threatened.push(c); }
        }

        // targets for land potential
        let mut srcs: Vec<(i64, i64)> = Vec::new();
        let mut any_land_target = false;
        for i in 0..nn as i64 {
            let t = &tiles[i as usize];
            if !t.known { srcs.push((i, 4)); continue; }
            if t.structure == 1 && t.owner != me {
                let occupied_by_me = unit_at[i as usize].map(|k| units[k].owner == me).unwrap_or(false);
                srcs.push((i, if t.owner > 0 { if t.capital != 0 { 0 } else { 0 } } else if occupied_by_me { 0 } else { 0 }));
            } else if t.structure == 2 { srcs.push((i, 1)); }
        }
        for e in &enemies { srcs.push((e.tile, 2)); }
        for &c in &threatened { if unit_at[c as usize].map(|k| units[k].owner != me).unwrap_or(true) { srcs.push((c, 0)); } }
        let climb = has(20);
        let land_pass = |t: i64| { let tl = &tiles[t as usize]; if !tl.known { return true; }
            match tl.terrain { 1 | 2 => tl.structure == 8 && tl.owner == me, 4 => climb, 0 => true, _ => true } };
        let srcs: Vec<(i64, i64)> = srcs.into_iter().filter(|&(t, c)| c == 4 || c == 2 || land_pass(t)).collect();
        let fland = field(n, &srcs, &land_pass);
        // reachable land targets from my units?
        for u in units.iter().filter(|u| u.owner == me && !naval(u.kind)) {
            if fland[u.tile as usize] < 1000 { any_land_target = true; }
        }
        let my_ports: Vec<i64> = (0..nn as i64).filter(|&i| tiles[i as usize].structure == 8 && tiles[i as usize].owner == me).collect();
        // naval potential: chebyshev-ish BFS through water/unknown to targets (land targets adjacent)
        let sea_pass = |t: i64| { let tl = &tiles[t as usize]; !tl.known || tl.terrain == 1 || tl.terrain == 2 };
        let mut nsrcs: Vec<(i64, i64)> = Vec::new();
        for i in 0..nn as i64 {
            let t = &tiles[i as usize];
            if t.structure == 1 && t.owner != me && t.known { nsrcs.push((i, if t.owner > 0 { 0 } else { 1 })); }
            if !t.known { nsrcs.push((i, 6)); }
        }
        for e in &enemies { nsrcs.push((e.tile, 3)); }
        let fsea = field(n, &nsrcs, &sea_pass);

        // fallback potential: go to ports
        let mut port_srcs: Vec<(i64, i64)> = my_ports.iter().filter(|&&p| fsea[p as usize] < 1000).map(|&p| (p, 0)).collect();
        if port_srcs.is_empty() { for &c in &my_cities { port_srcs.push((c, 3)); } }
        let fport = field(n, &port_srcs, &land_pass);
        let good_ports = my_ports.iter().filter(|&&p| fsea[p as usize] < 1000).count();
        let pot = |u: &Unit, t: i64| -> i64 {
            if naval(u.kind) { return fsea[t as usize]; }
            if any_land_target { fland[t as usize] } else { fport[t as usize] }
        };

        let want = 3 + 3 * ncity + if enemies.is_empty() { 0 } else { 3 };
        let summon_legal = legal.iter().any(|a| a["command"]["kind"].as_i64() == Some(4));
        let free_city = my_cities.iter().any(|&c| unit_at[c as usize].is_none());
        let mut need_army = mine_n < want && (summon_legal || (free_city && stars < 3));

        let research_plan: Vec<i64> = if !any_land_target && ncity > 0 && turn > 3 {
            vec![10, 1, 13, 6, 8, 20, 23, 2, 3, 44, 24, 16, 18, 4, 17, 14, 9, 7, 21, 19, 22, 12, 5]
        } else {
            vec![1, 20, 23, 24, 6, 8, 2, 3, 18, 16, 17, 10, 4, 13, 9, 7, 21, 19, 22, 44, 12, 14, 5]
        };

        let mut best: Option<(f64, &str)> = None;
        for a in legal {
            let c = &a["command"];
            let k = c["kind"].as_i64().unwrap_or(-1);
            let src = c["source"].as_i64().unwrap_or(-1);
            let dst = c["target"].as_i64().unwrap_or(-1);
            let p = c["parameter"].as_i64().unwrap_or(0);
            let id = a["action_id"].as_str().unwrap_or("");
            let mut s: f64 = match k {
                11 => -1e9,
                10 => 0.0,
                9 => (match p { 3 => 100.0, 4 => 60.0, 7 => 100.0, 1 => 50.0, 8 => 100.0, 5 => if !any_land_target && good_ports == 0 { 150.0 } else { 70.0 }, 6 => 100.0, 2 => 60.0, _ => 10.0 }) + 5000.0,
                8 => 3000.0 + if dst >= 0 && tiles[dst as usize].owner > 0 { 500.0 } else { 0.0 },
                2 => {
                    let au = unit_at.get(src as usize).and_then(|x| *x).map(|i| units[i].clone());
                    let du = unit_at.get(dst as usize).and_then(|x| *x).map(|i| units[i].clone());
                    match (au, du) {
                        (Some(au), Some(du)) => {
                            let (aa, _ad, ahp, arange, _) = stats(au.kind);
                            let (_da, dd, dhp, drange, dcost) = stats(du.kind);
                            let amax = ahp + if au.vet { 5.0 } else { 0.0 };
                            let dmax = dhp + if du.vet { 5.0 } else { 0.0 };
                            let dt = &tiles[dst as usize];
                            let bonus = if dt.structure == 1 { 1.5 } else if dt.terrain == 4 || dt.terrain == 5 { 1.5 } else { 1.0 };
                            let af = aa * au.hp / amax;
                            let df = dd * du.hp / dmax * bonus;
                            let dmg = (af / (af + df) * aa * 4.5).round();
                            let kill = dmg >= du.hp;
                            let dist = cheb(src, dst, n);
                            let ret = if kill || drange < dist { 0.0 } else { let df2 = dd * (du.hp - dmg) / dmax * bonus; (df2 / (af + df) * dd * 4.5).round() };
                            let _ = arange;
                            let mut sc = 2000.0 + dmg * 10.0 - ret * 8.0;
                            if kill { sc += 400.0 + dcost * 20.0; }
                            if ret >= au.hp { sc -= 1500.0; }
                            if !kill && dmg < ret && dmg < 3.0 { sc -= 1200.0; }
                            sc
                        }
                        _ => 1500.0,
                    }
                }
                3 => match p { 16 => 4000.0, 7 => {
                        let u = unit_at.get(src as usize).and_then(|x| *x).map(|i| units[i].clone());
                        match u { Some(u) => { let (_, _, mh, _, _) = stats(u.kind); if u.hp < mh * 0.5 { 900.0 } else { -10.0 } } None => -10.0 }
                    } _ => -1e6 },
                5 => 1200.0,
                6 => {
                    let base = match p { 5 => 800.0, 21 => 800.0, 12 => 700.0, 6 => 750.0, 13 => 750.0, 22 => 750.0, 8 => if dst < 0 || dst as usize >= nn || fsea[dst as usize] >= 1000 { -10.0 } else if !any_land_target && good_ports < 2 { 1300.0 } else if my_ports.len() < 1 && turn > 20 { 300.0 } else { -10.0 },
                        50 => 600.0, 17 | 18 | 19 | 20 => if stars > 25 { 200.0 } else { -10.0 }, 3 | 48 => -10.0, _ => -10.0 };
                    if need_army && base > 0.0 && !(p == 8) { base - 900.0 } else { base }
                }
                7 => {
                    let idx = research_plan.iter().position(|&t| t == p);
                    let mut r = match idx { Some(i) => 700.0 - (i as f64) * 20.0, None => 100.0 };
                    if turn < 2 && p == 1 { r = 1150.0; }
                    if need_army && !(p == 1 && enemies.is_empty()) && !(!any_land_target && (p == 10 || p == 13)) { r -= 1100.0; }
                    r
                }
                4 => {
                    let base = match p { 4 => 1100.0, 11 => 1080.0, 8 => 900.0, 3 => 1000.0, 9 => 960.0, 5 => if !threatened.is_empty() { 1090.0 } else { 500.0 }, 2 => if enemies.is_empty() { 950.0 } else { 1010.0 }, 42 => 1000.0, 44 => 1000.0, _ => 400.0 };
                    let mut b = base;
                    if !threatened.is_empty() { b += 600.0; }
                    if !need_army { b -= 500.0 }
                    if mine_n > 4 * ncity + 6 { b -= 600.0 }
                    b
                }
                1 => {
                    let u = unit_at.get(src as usize).and_then(|x| *x).map(|i| units[i].clone());
                    match u {
                        Some(u) => {
                            let ps = pot(&u, src); let pd = pot(&u, dst);
                            let mut sc = 100.0 + 15.0 * (ps.min(1000) - pd.min(1000)) as f64;
                            if pd >= ps { sc = -5.0 + (ps - pd).max(-5) as f64; }
                            // stay on capturable village
                            let st = &tiles[src as usize];
                            if st.structure == 1 && st.owner != me { sc = -100.0; }
                            // garrison threatened city
                            if st.structure == 1 && st.owner == me && enemies.iter().any(|e| cheb(e.tile, src, n) <= 3) {
                                if stars >= 3 && cheb(src, dst, n) == 1 && mine_n < want { sc = 20.0 + sc.max(0.0) * 0.1; } else { sc = -50.0; }
                            }
                            // leaving a city tile to free recruit slot is good
                            if st.structure == 1 && st.owner == me && sc < 0.0 && sc > -50.0 { sc = 5.0; }
                            // danger: count adjacent enemies at dst
                            let danger = enemies.iter().filter(|e| cheb(e.tile, dst, n) <= stats(e.kind).3.max(1)).count() as f64;
                            sc -= danger * 10.0 * if u.hp < 5.0 { 3.0 } else { 1.0 };
                            if naval(u.kind) && !naval_land(&tiles, dst) { if fsea[src as usize] >= 1000 { sc = 60.0; } else if pd > 0 { sc = -20.0; } }
                            sc
                        }
                        None => 1.0,
                    }
                }
                _ => -1000.0,
            };
            // budget: keep money for research when research desired is unaffordable? simple
            if k == 6 || k == 4 || k == 5 { if s < 0.0 { s = -10.0; } }
            if best.map(|b| s > b.0).unwrap_or(true) { best = Some((s, id)); }
        }
        let _ = is_water;
        best.map(|b| b.1.to_owned()).unwrap_or_else(|| legal[0]["action_id"].as_str().unwrap().to_owned())
    }
}

fn naval_land(tiles: &[Tile], t: i64) -> bool { let tr = tiles[t as usize].terrain; tr == 1 || tr == 2 }

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut bot = Bot { turn_seen: 0 };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = bot.choose(&obs);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
