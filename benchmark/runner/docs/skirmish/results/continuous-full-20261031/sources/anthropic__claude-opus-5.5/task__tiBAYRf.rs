use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

struct Mem { turn: i64, acts: u32 }

#[derive(Clone, Copy)]
struct U { kind: i64, owner: i64, hp: f64, maxhp: f64, moved: bool, attacked: bool }

fn stats(k: i64) -> (f64, f64, f64, f64, i64, i64) {
    // cost, health, attack, defense, move, range
    match k {
        2 => (2.0, 10.0, 2.0, 2.0, 1, 1),
        3 => (3.0, 10.0, 2.0, 1.0, 2, 1),
        4 => (8.0, 10.0, 3.5, 1.0, 3, 1),
        5 => (3.0, 15.0, 1.0, 3.0, 1, 1),
        8 => (8.0, 10.0, 4.0, 0.0, 1, 3),
        9 => (3.0, 10.0, 2.0, 1.0, 1, 2),
        10 => (5.0, 10.0, 0.0, 1.0, 1, 1),
        11 => (5.0, 15.0, 3.0, 3.0, 1, 1),
        12 => (10.0, 40.0, 5.0, 4.0, 1, 1),
        41 => (15.0, 10.0, 3.0, 2.0, 2, 3),
        42 => (5.0, 10.0, 2.0, 1.0, 3, 2),
        43 => (0.0, 10.0, 0.0, 1.0, 2, 2),
        44 => (5.0, 10.0, 3.0, 3.0, 3, 1),
        45 => (10.0, 40.0, 4.0, 4.0, 2, 1),
        39 => (2.0, 5.0, 0.0, 0.5, 2, 1),
        40 => (2.0, 10.0, 2.0, 2.0, 2, 1),
        _ => (3.0, 10.0, 2.0, 2.0, 1, 1),
    }
}
fn stiff(k: i64) -> bool { matches!(k, 8 | 10 | 43 | 41 | 45 | 39 | 17 | 22 | 25) }
fn naval(k: i64) -> bool { (39..=45).contains(&k) }
fn fortify(k: i64) -> bool { matches!(k, 2 | 3 | 4 | 5 | 9 | 20 | 25) }

fn research_rank(t: i64, stuck: bool) -> i64 {
    let order: &[i64] = if stuck {
        &[10, 13, 1, 6, 44, 2, 3, 16, 18, 20, 23, 8, 7, 24, 17, 14, 12, 4, 9, 21, 5, 19, 22, 38]
    } else {
        &[1, 6, 10, 20, 16, 2, 18, 23, 3, 8, 7, 24, 17, 4, 13, 9, 21, 5, 19, 22, 44, 14, 12, 38]
    };
    order.iter().position(|&x| x == t).map(|p| p as i64).unwrap_or(30)
}

fn combat(att: f64, ahp: f64, amax: f64, def: f64, dhp: f64, dmax: f64, bonus: f64) -> (f64, f64) {
    let af = att * ahp / amax;
    let df = def * dhp / dmax * bonus;
    if af + df <= 0.0 { return (0.0, 0.0); }
    let dmg = (af / (af + df) * att * 4.5).round();
    let ret = (df / (af + df) * def * 4.5).round();
    (dmg, ret)
}

fn choose(obs: &Value, mem: &mut Mem) -> String {
    let legal = obs["legal_actions"].as_array().expect("legal_actions");
    let me = obs["player"].as_i64().unwrap_or(1);
    let vs = &obs["visible_state"];
    let size = vs["board_size"].as_i64().unwrap_or(14) as usize;
    let n = size * size;
    let turn = vs["turn"].as_i64().unwrap_or(0);
    if turn != mem.turn { mem.turn = turn; mem.acts = 0; }
    mem.acts += 1;
    let stars = vs["stars"].as_f64().unwrap_or(0.0);
    let techs: Vec<i64> = vs["technology"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default();
    let has = |t: i64| techs.contains(&t);

    let mut terrain = vec![-1i64; n];
    let mut structure = vec![0i64; n];
    let mut towner = vec![0i64; n];
    let mut resource = vec![0i64; n];
    if let Some(ts) = vs["tiles"].as_array() {
        for t in ts {
            let i = t["tile"].as_i64().unwrap_or(-1);
            if i < 0 || i as usize >= n { continue; }
            let i = i as usize;
            if t["known"].as_bool().unwrap_or(false) {
                terrain[i] = t["terrain"].as_i64().unwrap_or(0);
                structure[i] = t["structure"].as_i64().unwrap_or(0);
                towner[i] = t["owner"].as_i64().unwrap_or(0);
                resource[i] = t["resource"].as_i64().unwrap_or(0);
            }
        }
    }
    let mut units: Vec<Option<U>> = vec![None; n];
    let mut my_units = 0usize;
    let mut enemies: Vec<usize> = vec![];
    let mut homecnt = vec![0i64; n];
    if let Some(us) = vs["units"].as_array() {
        for u in us {
            let i = u["tile"].as_i64().unwrap_or(-1);
            if i < 0 || i as usize >= n { continue; }
            let kind = u["kind"].as_i64().unwrap_or(2);
            let owner = u["owner"].as_i64().unwrap_or(0);
            let mut maxhp = stats(kind).1;
            if u["veteran"].as_bool().unwrap_or(false) { maxhp += 5.0; }
            let uu = U { kind, owner, hp: u["health"].as_f64().unwrap_or(10.0), maxhp,
                moved: u["moved"].as_bool().unwrap_or(false), attacked: u["attacked"].as_bool().unwrap_or(false) };
            units[i as usize] = Some(uu);
            if owner == me { my_units += 1; let h = u["home"].as_i64().unwrap_or(-1); if h >= 0 && (h as usize) < n { homecnt[h as usize] += 1; } } else { enemies.push(i as usize); }
        }
    }
    let mut my_city = vec![false; n];
    let mut enemy_city = vec![false; n];
    let mut ncities = 0usize;
    let mut city_list = vec![];
    let mut clevel = vec![0i64; n];
    if let Some(cs) = vs["cities"].as_array() {
        for c in cs {
            let i = c["tile"].as_i64().unwrap_or(-1);
            if i < 0 || i as usize >= n { continue; }
            let o = c["owner"].as_i64().unwrap_or(0);
            if o == me { my_city[i as usize] = true; ncities += 1; city_list.push(i as usize); clevel[i as usize] = c["level"].as_i64().unwrap_or(1); }
            else if o != 0 { enemy_city[i as usize] = true; }
        }
    }
    let xy = |t: usize| ((t % size) as i64, (t / size) as i64);
    let dist = |a: usize, b: usize| { let (ax, ay) = xy(a); let (bx, by) = xy(b); (ax - bx).abs().max((ay - by).abs()) };
    let nbrs = |t: usize| {
        let (x, y) = xy(t);
        let mut v = Vec::with_capacity(8);
        for dy in -1..=1i64 { for dx in -1..=1i64 {
            if dx == 0 && dy == 0 { continue; }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < size && (ny as usize) < size { v.push(ny as usize * size + nx as usize); }
        }}
        v
    };
    // threat: enemy strength near tile
    let mut threat = vec![0.0f64; n];
    for &e in &enemies {
        let u = units[e].unwrap();
        let (_, _, a, _, mv, rg) = stats(u.kind);
        let reach = mv + rg;
        for t in 0..n { if dist(e, t) <= reach { threat[t] += a * u.hp / u.maxhp; } }
    }
    // capture targets
    let mut capt = vec![false; n];
    for t in 0..n { if structure[t] == 1 && !my_city[t] { capt[t] = true; } }
    if enemy_city.iter().any(|&b| b) { for t in 0..n { if enemy_city[t] { capt[t] = true; } } }

    let land_ok = |t: usize| -> bool {
        let tr = terrain[t];
        tr == -1 || tr == 3 || tr == 5 || tr == 6 || (tr == 4 && has(20))
    };
    let water_ok = |t: usize| -> bool { let tr = terrain[t]; tr == -1 || tr == 1 || tr == 2 || (structure[t] == 8) };
    // source offsets
    let mut src: Vec<(usize, i64)> = vec![];
    for t in 0..n {
        if capt[t] { src.push((t, if enemy_city[t] { 0 } else { 0 })); }
        else if structure[t] == 2 { src.push((t, 2)); }
    }
    for &e in &enemies { src.push((e, 2)); }
    let mut guess: Vec<usize> = vec![];
    if !enemy_city.iter().any(|&b| b) {
        for &c in &city_list {
            let (x, y) = xy(c);
            let g = (size as i64 - 1 - y) as usize * size + (size as i64 - 1 - x) as usize;
            if terrain[g] == -1 || (structure[g] == 1 && !my_city[g]) { guess.push(g); }
        }
        if guess.is_empty() { guess.push(n / 2 + size / 2); }
    } else {
        for t in 0..n { if enemy_city[t] { guess.push(t); } }
    }
    let gsrc: Vec<(usize, i64)> = guess.iter().map(|&g| (g, 0)).collect();
    for &g in &guess { src.push((g, 3)); }
    let bfs = |srcs: &Vec<(usize, i64)>, ok: &dyn Fn(usize) -> bool| -> Vec<i64> {
        let mut d = vec![i64::MAX / 4; n];
        // dial-like: small offsets, use repeated relax via deque sorted by buckets
        let mut buckets: Vec<Vec<usize>> = vec![vec![]; 8];
        for &(t, o) in srcs { if o < d[t] { d[t] = o; buckets[o as usize].push(t); } }
        let mut q: VecDeque<usize> = VecDeque::new();
        // simple: process buckets in order with BFS layering
        let mut layer = 0usize;
        let mut cur: Vec<usize> = vec![];
        loop {
            if layer < buckets.len() { for &t in &buckets[layer] { if d[t] == layer as i64 { cur.push(t); } } }
            if cur.is_empty() && layer >= buckets.len() { break; }
            let mut next = vec![];
            for &t in &cur {
                if d[t] != layer as i64 { continue; }
                for nb in nbrs(t) {
                    if !ok(nb) { continue; }
                    if d[nb] > layer as i64 + 1 { d[nb] = layer as i64 + 1; next.push(nb); }
                }
            }
            q.clear();
            cur = next;
            layer += 1;
            if layer > n + 10 { break; }
        }
        d
    };
    let lsrc: Vec<(usize, i64)> = src.iter().cloned().filter(|&(t, _)| land_ok(t)).collect();
    let dland = bfs(&lsrc, &land_ok);
    let dwater = bfs(&src, &water_ok);
    let any_ok = |_t: usize| true;
    let dguess = bfs(&gsrc, &any_ok);
    // home field for retreat
    let home_src: Vec<(usize, i64)> = city_list.iter().map(|&c| (c, 0)).collect();
    let dhome = bfs(&home_src, &land_ok);
    // stuck: no land target reachable from any of our cities
    let inf = i64::MAX / 8;
    let stuck = turn >= 4 && city_list.iter().all(|&c| dland[c] >= inf);
    let have_port = (0..n).any(|t| structure[t] == 8 && towner[t] == me);
    let ports: Vec<usize> = (0..n).filter(|&t| structure[t] == 8).collect();

    let mut best: Option<(f64, &Value)> = None;
    for a in legal {
        let c = &a["command"];
        let k = c["kind"].as_i64().unwrap_or(-1);
        let s = c["source"].as_i64().unwrap_or(-1);
        let t = c["target"].as_i64().unwrap_or(-1);
        let p = c["parameter"].as_i64().unwrap_or(0);
        let su = if s >= 0 && (s as usize) < n { units[s as usize] } else { None };
        let tu = if t >= 0 && (t as usize) < n { units[t as usize] } else { None };
        let score: f64 = match k {
            9 => 10000.0 + match p { 3 => 10.0, 4 => 5.0, 7 => 10.0, 1 => if t >= 0 && (t as usize) < n && threat[t as usize] > 3.0 { 12.0 } else { 5.0 },
                8 => 10.0, 5 => 6.0, 6 => 10.0, 2 => 5.0, _ => 0.0 },
            8 => 9000.0,
            2 => {
                if let (Some(a1), Some(d1)) = (su, tu) {
                    let (_, _, att, _, _, _) = stats(a1.kind);
                    let (dc, _, _, def, _, drg) = stats(d1.kind);
                    let tt = t as usize;
                    let mut bonus: f64 = 1.0;
                    if (enemy_city[tt] || (structure[tt] == 1 && !my_city[tt])) && fortify(d1.kind) { bonus = 1.5; }
                    if terrain[tt] == 4 || terrain[tt] == 5 { bonus = bonus.max(1.2); }
                    let (dmg, mut ret) = combat(att, a1.hp, a1.maxhp, def, d1.hp, d1.maxhp, bonus);
                    if stiff(d1.kind) || dist(s as usize, tt) > drg { ret = 0.0; }
                    if dmg >= d1.hp { 2000.0 + dc * 30.0 }
                    else if ret >= a1.hp { -1.0 }
                    else { let v = dmg * 1.0 - ret * 0.8; if v > 0.0 { 600.0 + v * 30.0 } else { -1.0 } }
                } else { 500.0 }
            }
            1 => {
                if let Some(u) = su {
                    let (ss, tt) = (s as usize, t as usize);
                    let nav = naval(u.kind);
                    let df = if nav { &dwater } else { &dland };
                    let ds = df[ss].min(inf); let dt = df[tt].min(inf);
                    let mut sc = if ds >= inf && dt >= inf { 0.0 } else { (ds - dt) as f64 * 30.0 };
                    sc += (dguess[ss].min(inf) - dguess[tt].min(inf)) as f64 * 10.0;
                    let unk = nbrs(tt).iter().filter(|&&q| terrain[q] == -1).count() as f64;
                    sc += unk * 6.0;
                    if capt[tt] { sc += 400.0; }
                    if structure[tt] == 2 { sc += 150.0; }
                    // embark onto port when stuck
                    if !nav && stuck && structure[tt] == 8 { sc += 300.0; }
                    if !nav && stuck && !ports.is_empty() {
                        let pd = |x: usize| ports.iter().map(|&q| dist(x, q)).min().unwrap_or(99);
                        sc += (pd(ss) - pd(tt)) as f64 * 40.0;
                    }
                    let ratio = u.hp / u.maxhp;
                    if ratio < 0.45 && threat[ss] > 0.0 {
                        sc = (dhome[ss].min(inf) - dhome[tt].min(inf)) as f64 * 30.0 - threat[tt] * 10.0;
                    }
                    // garrison
                    let can_recruit = my_city[ss] && homecnt[ss] <= clevel[ss] && stars >= 3.0;
                    if my_city[ss] && u.kind == 5 { sc -= 1000.0; }
                    else if my_city[ss] && threat[ss] >= 1.5 {
                        if can_recruit && threat[tt] < 6.0 { sc += 150.0; } else { sc -= 500.0; }
                    } else if can_recruit { sc += 60.0; }
                    if u.kind == 5 && !my_city[ss] && my_city[tt] { sc += 300.0; }
                    if capt[ss] && !my_city[ss] { sc -= 300.0; }
                    // danger
                    let (_, _, _, def, _, _) = stats(u.kind);
                    let danger = threat[tt] - def * ratio * 1.5;
                    if danger > 0.0 { sc -= danger * 12.0; }
                    if sc > 0.0 { 100.0 + sc } else { -1.0 }
                } else { -1.0 }
            }
            5 => 450.0,
            6 => match p {
                5 | 21 => 380.0, 6 | 13 | 22 => 400.0, 12 => 300.0, 50 => 300.0,
                8 => if stuck && !have_port { 900.0 } else if stars >= 25.0 { 150.0 } else { -1.0 },
                23..=29 => 600.0,
                17 | 18 | 19 | 20 => if stars >= 22.0 { 280.0 } else { -1.0 },
                3 => if stars >= 40.0 { 60.0 } else { -1.0 },
                32 | 33 | 41 => if stars >= 30.0 { 100.0 } else { -1.0 },
                _ => -1.0,
            },
            7 => {
                let cnt = |f: &dyn Fn(usize) -> bool| (0..n).filter(|&q| towner[q] == me && f(q)).count() as f64;
                let v: f64 = match p {
                    1 => 400.0,
                    6 => if cnt(&|q| resource[q] == 6) >= 1.0 { 390.0 } else { 250.0 },
                    10 => if stuck { 420.0 } else if cnt(&|q| resource[q] == 3) >= 1.0 { 385.0 } else { 200.0 },
                    16 => if cnt(&|q| terrain[q] == 5) >= 2.0 { 360.0 } else { 220.0 },
                    20 => if cnt(&|q| resource[q] == 5) >= 1.0 { 340.0 } else { 150.0 },
                    23 => if cnt(&|q| resource[q] == 5) >= 1.0 { 350.0 } else { 100.0 },
                    8 => if cnt(&|q| resource[q] == 2) >= 1.0 { 330.0 } else { 150.0 },
                    13 => if stuck { 410.0 } else { 120.0 },
                    44 => if stuck { 380.0 } else { 100.0 },
                    2 => 300.0, 3 => 310.0, 18 => 280.0, 7 => 375.0, 24 => 260.0, 17 => 250.0,
                    9 => 200.0, 4 => 150.0, 5 => 150.0, 21 => 140.0, 19 => 150.0, 22 => 120.0,
                    14 => if stuck { 300.0 } else { 90.0 }, 12 => 90.0,
                    _ => if stars >= 25.0 { 80.0 } else { -1.0 },
                };
                let _ = research_rank(p, stuck);
                let mil = matches!(p, 2 | 3 | 18 | 24 | 17 | 9 | 4 | 5 | 21 | 19 | 22);
                if v < 0.0 { v }
                else if stars >= 20.0 { v + 150.0 }
                else if (p == 1 && turn >= 2) || (p == 7 && turn >= 5) { v + 30.0 } else if mil && ncities < 3 { v - 200.0 } else { v - 50.0 }
            }
            4 => {
                if su.map(|u| naval(u.kind)).unwrap_or(false) {
                    // naval upgrade
                    match p { 42 => 500.0, 44 => 520.0, 45 => 300.0, 41 => 200.0, _ => 100.0 }
                } else {
                    let tt = s.max(0) as usize;
                    let want = 99usize;
                    let threatened = threat[tt] >= 1.5;
                    let pref = match p {
                        4 => 10.0, 11 => 8.0, 8 => if my_units >= 4 { 6.0 } else { 1.0 }, 9 => if threatened { 7.0 } else { 5.0 },
                        3 => if turn < 25 { 6.0 } else { 4.0 }, 5 => if threatened { 5.5 } else { 3.0 }, 2 => if threatened { 6.5 } else { 2.0 }, _ => -100.0 };
                    if pref < 0.0 { -1.0 }
                    else if threatened { 950.0 + pref }
                    else if my_units < want { 440.0 + pref * 3.0 }
                    else { -1.0 }
                }
            }
            3 => match p {
                16 => 1500.0,
                7 => if let Some(u) = su { let r = u.hp / u.maxhp; if r < 0.95 { (1.0 - r) * 400.0 } else { -1.0 } } else { -1.0 },
                _ => -1.0,
            },
            10 => 0.0,
            _ => -5.0,
        };
        let score = if mem.acts > 300 && k != 10 && k != 9 { -10.0 } else { score };
        let tie = a["action_id"].as_str().unwrap_or("");
        let _ = tie;
        match best { Some((b, _)) if b >= score => {}, _ => best = Some((score, a)) }
    }
    let _ = (towner.len(), my_units);
    let a = best.map(|x| x.1).unwrap_or(&legal[0]);
    a["action_id"].as_str().unwrap().to_owned()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { turn: -1, acts: 0 };
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let id = choose(&obs, &mut mem);
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
