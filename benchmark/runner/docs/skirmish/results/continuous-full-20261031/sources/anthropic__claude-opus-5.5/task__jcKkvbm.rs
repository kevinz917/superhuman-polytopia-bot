use serde_json::{json, Value};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;
use std::io::{self, BufRead, Write};

// (cost, hp, atk, def, move, range)
fn stats(k: i64) -> (f64, f64, f64, f64, i64, i64) {
    match k {
        2 => (2., 10., 2., 2., 1, 1),
        3 => (3., 10., 2., 1., 2, 1),
        4 => (8., 10., 3.5, 1., 3, 1),
        5 => (3., 15., 1., 3., 1, 1),
        8 => (8., 10., 4., 0., 1, 3),
        9 => (3., 10., 2., 1., 1, 2),
        10 => (5., 10., 0., 1., 1, 1),
        11 => (5., 15., 3., 3., 1, 1),
        12 => (10., 40., 5., 4., 1, 1),
        15 => (2., 15., 3., 1., 1, 1),
        39 => (2., 5., 0., 0.5, 2, 1),
        40 => (2., 10., 2., 2., 2, 1),
        41 => (15., 10., 3., 2., 2, 3),
        42 => (5., 10., 2., 1., 3, 2),
        43 => (0., 10., 0., 1., 2, 2),
        44 => (5., 10., 3., 3., 3, 1),
        45 => (10., 40., 4., 4., 2, 1),
        _ => (3., 10., 2., 2., 1, 1),
    }
}

#[derive(Clone)]
struct Unit { kind: i64, owner: i64, hp: f64, vet: bool, moved: bool, attacked: bool, passenger: i64 }

struct Mem { turn_seen: i64, enemy_capital: Option<usize> }

fn gi(v: &Value, k: &str) -> i64 { v[k].as_i64().unwrap_or(0) }

fn choose(obs: &Value, mem: &mut Mem) -> String {
    let legal = obs["legal_actions"].as_array().expect("legal_actions");
    let me = gi(obs, "player");
    let vs = &obs["visible_state"];
    let n = vs["board_size"].as_u64().unwrap_or(14) as usize;
    let nn = n * n;
    let stars = gi(vs, "stars");
    let turn = gi(vs, "turn");
    mem.turn_seen = turn;
    let techs: HashSet<i64> = vs["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
    let mut terrain = vec![-1i64; nn];
    let mut structure = vec![0i64; nn];
    let mut towner = vec![0i64; nn];
    let mut known = vec![false; nn];
    if let Some(ts) = vs["tiles"].as_array() {
        for t in ts {
            let i = gi(t, "tile") as usize;
            if i >= nn { continue; }
            if t["known"].as_bool().unwrap_or(false) {
                known[i] = true;
                terrain[i] = gi(t, "terrain");
                structure[i] = gi(t, "structure");
                towner[i] = gi(t, "owner");
            }
        }
    }
    let mut units: HashMap<usize, Unit> = HashMap::new();
    if let Some(us) = vs["units"].as_array() {
        for u in us {
            let i = gi(u, "tile") as usize;
            units.insert(i, Unit {
                kind: gi(u, "kind"), owner: gi(u, "owner"), hp: u["health"].as_f64().unwrap_or(10.),
                vet: u["veteran"].as_bool().unwrap_or(false), moved: u["moved"].as_bool().unwrap_or(false),
                attacked: u["attacked"].as_bool().unwrap_or(false), passenger: gi(u, "passenger"),
            });
        }
    }
    let mut my_cities: Vec<usize> = vec![];
    let mut enemy_cities: Vec<usize> = vec![];
    let mut city_level: HashMap<usize, i64> = HashMap::new();
    if let Some(cs) = vs["cities"].as_array() {
        for c in cs {
            let t = gi(c, "tile") as usize;
            city_level.insert(t, gi(c, "level"));
            if gi(c, "owner") == me { my_cities.push(t) } else if gi(c, "owner") != 0 { enemy_cities.push(t) }
        }
    }
    let my_city_set: HashSet<usize> = my_cities.iter().cloned().collect();
    let xy = |t: usize| ((t % n) as i64, (t / n) as i64);
    let dist = |a: usize, b: usize| { let (ax, ay) = xy(a); let (bx, by) = xy(b); (ax - bx).abs().max((ay - by).abs()) };
    // villages (non-mine)
    let mut villages: Vec<usize> = vec![];
    for i in 0..nn { if structure[i] == 1 && !my_city_set.contains(&i) { villages.push(i); } }
    for &c in &enemy_cities { if !villages.contains(&c) { villages.push(c); } }
    let enemy_units: Vec<usize> = units.iter().filter(|(_, u)| u.owner != me && u.owner != 0).map(|(t, _)| *t).collect();
    let my_units: Vec<usize> = units.iter().filter(|(_, u)| u.owner == me).map(|(t, _)| *t).collect();
    // enemy capital guess
    if mem.enemy_capital.is_none() {
        if let Some(&c) = my_cities.first() { let (x, y) = xy(c); let g = ((n as i64 - 1 - y) * n as i64 + (n as i64 - 1 - x)) as usize; mem.enemy_capital = Some(g); }
    }
    let water_ok = techs.contains(&10) || my_units.iter().any(|t| terrain[*t] == 1 || terrain[*t] == 2);
    // Dijkstra field
    let cost_of = |i: usize, land_only: bool| -> i64 {
        match terrain[i] {
            -1 => 2,
            1 => if land_only { 1000 } else if water_ok { 2 } else { 6 },
            2 => if land_only { 1000 } else if water_ok { 3 } else { 8 },
            4 => if techs.contains(&20) { 2 } else { 60 },
            5 => 2,
            _ => 1,
        }
    };
    let occupied_target = |t: usize| -> bool {
        if let Some(u) = units.get(&t) { if u.owner == me { return true; } }
        false
    };
    let field = |land_only: bool, military: bool, only_v: bool| -> Vec<i64> {
        let mut d = vec![i64::MAX / 4; nn];
        let mut h = BinaryHeap::new();
        let mut seed = |t: usize, v: i64, d: &mut Vec<i64>, h: &mut BinaryHeap<Reverse<(i64, usize)>>| { if v < d[t] { d[t] = v; h.push(Reverse((v, t))); } };
        for &v in &villages { if !occupied_target(v) { let base = if enemy_cities.contains(&v) { if military { 0 } else { 4 } } else { 0 }; seed(v, base * 10, &mut d, &mut h); } }
        if military && !only_v { for &e in &enemy_units { seed(e, 30, &mut d, &mut h); } }
        for i in 0..nn { if only_v { break; } if !known[i] { let hd = my_cities.iter().map(|c| dist(*c, i)).min().unwrap_or(0); seed(i, 40 + 4 * hd, &mut d, &mut h); } else if structure[i] == 2 { seed(i, 15, &mut d, &mut h); } }
        if let Some(g) = mem.enemy_capital { if !known[g] && !only_v { seed(g, 20, &mut d, &mut h); } }
        while let Some(Reverse((v, t))) = h.pop() {
            if v > d[t] { continue; }
            let (x, y) = xy(t);
            for dy in -1..=1 { for dx in -1..=1 {
                if dx == 0 && dy == 0 { continue; }
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= n as i64 || ny >= n as i64 { continue; }
                let j = (ny * n as i64 + nx) as usize;
                let nv = v + 10 * cost_of(t, land_only);
                if nv < d[j] { d[j] = nv; h.push(Reverse((nv, j))); }
            }}
        }
        d
    };
    let fmil = field(false, true, false);
    let fland = field(true, true, false);
    let fvl = field(true, true, true);
    const STUCK: i64 = 5000;
    let stuck = |t: usize| -> bool { terrain[t] >= 3 && fvl[t] >= STUCK && fland[t] >= STUCK };
    let stuck_units: Vec<usize> = my_units.iter().cloned().filter(|t| stuck(*t)).collect();
    let rich = stars > 30;
    // threat near cities
    let threat = |c: usize| -> i64 { enemy_units.iter().filter(|e| dist(**e, c) <= 3).count() as i64 };
    let num_mil = my_units.len() as i64;
    let desired = (3 + 2 * my_cities.len() as i64).max(enemy_units.len() as i64 + 2).min(24);
    let tech_order: Vec<i64> = {
        let mut v = vec![];
        let watery = (0..nn).filter(|i| terrain[*i] == 1 || terrain[*i] == 2).count() * 3 > (0..nn).filter(|i| known[*i]).count().max(1);
        if watery { v.push(10); }
        v.extend_from_slice(&[1, 6, 18, 20, 23, 24, 2, 3, 8, 7, 10, 16, 17, 9, 4, 13, 21, 19, 44, 22, 5, 12, 14]);
        v
    };
    let next_tech = tech_order.iter().position(|t| !techs.contains(t)).map(|p| tech_order[p]);
    let can_afford_next = legal.iter().any(|a| a["command"]["kind"] == 7 && Some(gi(&a["command"], "parameter")) == next_tech);
    let any_port = (0..nn).any(|i| structure[i] == 8 && towner[i] == me);
    let land_stuck = my_units.iter().all(|t| fland[*t] >= 1000 * 10 / 2) && turn > 3;
    let summon_legal = legal.iter().any(|a| a["command"]["kind"] == 4);
    let max_hp = |u: &Unit| stats(u.kind).1 + if u.vet { 5. } else { 0. };

    let mut best: Option<(f64, &str)> = None;
    for a in legal {
        let c = &a["command"];
        let k = gi(c, "kind");
        let src = gi(c, "source");
        let dst = gi(c, "target");
        let param = gi(c, "parameter");
        let score: f64 = match k {
            11 => -1e9,
            10 => 0.0,
            9 => 1000.0 + match param { 3 => 5., 7 => 5., 8 => 5., 6 => 5., 1 => if threat(dst.max(0) as usize) > 0 { 8. } else { 1. }, 2 => 1., _ => 0. },
            8 => 900.0,
            2 => {
                let (s, d) = (src as usize, dst as usize);
                match (units.get(&s), units.get(&d)) {
                    (Some(au), Some(du)) => {
                        let (_, _, aatk, _, _, _) = stats(au.kind);
                        let (dcost, _, _, ddef, _, drange) = stats(du.kind);
                        let mut bonus = 1.0;
                        if structure[d] == 1 { bonus = 1.5; }
                        if terrain[d] == 4 || terrain[d] == 5 { bonus = 1.5; }
                        let af = aatk * au.hp / max_hp(au);
                        let df = ddef * du.hp / max_hp(du) * bonus;
                        let dmg = (af / (af + df) * aatk * 4.5).round();
                        let kill = dmg >= du.hp;
                        let mut sc = 300.0;
                        if kill { sc += 60. + 10. * dcost + if structure[d] == 1 { 40. } else { 0. }; }
                        else {
                            sc += dmg * 4.;
                            if dist(s, d) <= drange {
                                let ndf = ddef * (du.hp - dmg) / max_hp(du) * bonus;
                                let ret = (ndf / (af + ndf) * ddef * 4.5).round();
                                if ret >= au.hp { sc -= 60. + 10. * stats(au.kind).0; } else { sc -= ret * 3.5; }
                            }
                        }
                        if my_cities.iter().any(|c| dist(*c, d) <= 2) { sc += 20.; }
                        if sc < 285. { -1.0 } else { sc }
                    }
                    _ => 250.0,
                }
            }
            3 => match param { 16 => 950.0, 7 => {
                let s = src as usize;
                if let Some(u) = units.get(&s) { if u.hp < max_hp(u) * 0.4 && !enemy_units.iter().any(|e| dist(*e, s) <= 2) && towner[s] == me { 60.0 } else if u.hp < max_hp(u) * 0.5 && !enemy_units.iter().any(|e| dist(*e, s) <= 1) { 20.0 } else { -1.0 } } else { -1.0 }
            }, _ => -1.0 },
            5 => 200.0,
            7 => {
                if Some(param) == next_tech { let tier = match param { 1|6|10|15|20 => 1, 3|5|9|12|14|17|19|22|24 => 3, _ => 2 }; let cost = 4 + my_cities.len() as i64 * tier; if (!summon_legal && (stars >= cost + 3 || enemy_units.is_empty())) || rich || num_mil >= 28 || stars >= cost + 12 { let _ = desired; 26.0 } else { -1.0 } } else if matches!(param, 18 | 22 | 25 | 26 | 30 | 31 | 32 | 35 | 36 | 37 | 41 | 42 | 49 | 120 | 121) { -1.0 } else if rich { 40.0 } else { -1.0 }
            }
            4 => {
                let city = src as usize;
                let th = threat(city);
                let narch = my_units.iter().filter(|t| units[*t].kind == 9).count() as i64; let nmel = num_mil - narch;
                let ndef = my_units.iter().filter(|t| units[*t].kind == 5).count() as i64;
                let pref = match param { 4 => 40., 11 => 35., 12 => 45., 3 => if turn < 20 { 26. } else { 16. }, 9 => if narch * 3 < nmel { 28. } else { 12. }, 8 => if narch > 0 { 18. } else { 8. }, 5 => if th > 1 && ndef < my_cities.len() as i64 { 30. } else { 3. }, 2 => if turn < 12 { 20. } else { 5. }, 40 | 44 | 42 => 10., _ => -100. };
                if num_mil >= 28 { -1.0 }
                else if num_mil < desired || th > 0 { 170.0 + pref }
                                else { 100.0 + pref }
            }
            6 => {
                let t = dst as usize;
                match param {
                    8 => { let _ = any_port; let _ = land_stuck; if stuck_units.iter().any(|u| dist(*u, t) <= 2 && !(0..nn).any(|q| structure[q] == 8 && towner[q] == me && dist(q, *u) <= 4)) { 190.0 } else if rich { 45.0 } else { -1.0 } },
                    5 | 21 => 29.0,
                    12 => 28.0,
                    6 | 13 | 22 => 27.0,
                    3 | 48 => -1.0,
                    _ => if rich { 50.0 } else { -1.0 },
                }
            }
            1 => {
                let (s, d) = (src as usize, dst as usize);
                let u = match units.get(&s) { Some(u) => u, None => continue };
                // don't leave a village being captured
                if structure[s] == 1 && !my_city_set.contains(&s) && !u.moved { -1.0 }
                else if my_city_set.contains(&s) && threat(s) > 0 && !units.get(&s).map(|u| u.passenger != 0).unwrap_or(false) { -1.0 }
                else {
                    let f = if terrain[s] >= 3 && !stuck(s) { &fland } else { &fmil };
                    let gain = (f[s] - f[d]).min(200) as f64 / 10.0;
                    let mut sc = 30.0 + 12.0 * gain;
                    if terrain[s] >= 3 && stuck(s) && structure[d] == 8 { sc += 100.; }
                    // defend threatened empty city
                    if my_city_set.contains(&d) && threat(d) > 0 { sc += 60.; }
                    // danger: count adjacent enemies at dest
                    let adj = enemy_units.iter().filter(|e| dist(**e, d) <= 1).count() as f64;
                    if u.hp < max_hp(u) * 0.4 { sc -= adj * 15.; }
                    let mut incoming = 0.0;
                    let (_, _, _, mydef, _, _) = stats(u.kind);
                    let bonus = if structure[d] == 1 && my_city_set.contains(&d) { 1.5 } else if terrain[d] == 4 || terrain[d] == 5 { 1.3 } else { 1.0 };
                    for e in &enemy_units { let eu = &units[e]; let (_, _, eatk, _, emv, erng) = stats(eu.kind); if dist(*e, d) as i64 <= emv.min(2) + erng - 1 + if emv >= 2 {1} else {0} { let af = eatk * eu.hp / max_hp(eu); let df = mydef * u.hp / max_hp(u) * bonus; incoming += (af / (af + df) * eatk * 4.5).round(); } }
                    let target_tile = structure[d] == 1 && !my_city_set.contains(&d);
                    if incoming >= u.hp && !target_tile { sc -= 25. + 4. * stats(u.kind).0; }
                    let _ = u.attacked;
                    if gain <= 0.0 && sc < 100. { -1.0 } else { sc }
                }
            }
            _ => -1.0,
        };
        let id = a["action_id"].as_str().unwrap_or("");
        if best.map(|(b, _)| score > b).unwrap_or(true) { best = Some((score, id)); }
    }
    if best.is_none() { return legal.get(0).and_then(|a| a["action_id"].as_str()).unwrap_or("0").to_owned(); }
    let (_, id) = best.unwrap();
    // prefer end turn on non-positive
    if best.unwrap().0 <= 0.0 {
        if let Some(e) = legal.iter().find(|a| a["command"]["kind"] == 10) { return e["action_id"].as_str().unwrap().to_owned(); }
    }
    id.to_owned()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { turn_seen: 0, enemy_capital: None };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = choose(&obs, &mut mem);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
