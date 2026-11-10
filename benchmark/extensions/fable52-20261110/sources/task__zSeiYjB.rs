use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn gi(v: &Value) -> i64 {
    v.as_i64().unwrap_or(-1)
}

struct Mem {
    turn: i64,
    count: i64,
}

fn choose(obs: &Value, mem: &mut Mem) -> String {
    let empty: Vec<Value> = Vec::new();
    let legal = obs["legal_actions"].as_array().unwrap_or(&empty);
    if legal.is_empty() {
        return String::new();
    }
    let vs = &obs["visible_state"];
    let me = gi(&obs["player"]);
    let size = vs["board_size"].as_i64().unwrap_or(14).max(1);
    let n = (size * size) as usize;
    let turn = gi(&vs["turn"]);
    if turn != mem.turn {
        mem.turn = turn;
        mem.count = 0;
    }
    mem.count += 1;
    let id = |a: &Value| a["action_id"].as_str().unwrap_or("").to_owned();
    let kind = |a: &Value| gi(&a["command"]["kind"]);
    let par = |a: &Value| gi(&a["command"]["parameter"]);
    let src = |a: &Value| gi(&a["command"]["source"]);
    let tgt = |a: &Value| gi(&a["command"]["target"]);
    let dist = |a: i64, b: i64| -> i64 { ((a % size) - (b % size)).abs().max(((a / size) - (b / size)).abs()) };
    let end = legal.iter().find(|a| kind(a) == 10);
    let fallback = || -> String {
        if let Some(e) = end {
            return id(e);
        }
        if let Some(a) = legal.iter().find(|a| kind(a) != 11 && !(kind(a) == 3 && par(a) >= 18)) {
            return id(a);
        }
        id(&legal[0])
    };
    // rewards
    let rewards: Vec<&Value> = legal.iter().filter(|a| kind(a) == 9).collect();
    if !rewards.is_empty() {
        let pref = [3i64, 7, 8, 6, 5, 2, 1, 4];
        for p in pref.iter() {
            if let Some(a) = rewards.iter().find(|a| par(a) == *p) {
                return id(a);
            }
        }
        return id(rewards[0]);
    }
    if mem.count > 150 {
        return fallback();
    }
    // board
    let mut known = vec![false; n];
    let mut structure = vec![0i64; n];
    let mut towner = vec![0i64; n];
    if let Some(tiles) = vs["tiles"].as_array() {
        for t in tiles {
            let i = gi(&t["tile"]);
            if i < 0 || i as usize >= n {
                continue;
            }
            let i = i as usize;
            known[i] = t["known"].as_bool().unwrap_or(false);
            if known[i] {
                structure[i] = t["structure"].as_i64().unwrap_or(0);
                towner[i] = t["owner"].as_i64().unwrap_or(0);
            }
        }
    }
    let mut uowner = vec![0i64; n];
    let mut uhealth = vec![0f64; n];
    let mut my_units = 0i64;
    if let Some(units) = vs["units"].as_array() {
        for u in units {
            let i = gi(&u["tile"]);
            if i < 0 || i as usize >= n {
                continue;
            }
            uowner[i as usize] = gi(&u["owner"]);
            uhealth[i as usize] = u["health"].as_f64().unwrap_or(10.0);
            if gi(&u["owner"]) == me {
                my_units += 1;
            }
        }
    }
    let mut my_cities = 0i64;
    if let Some(cs) = vs["cities"].as_array() {
        for c in cs {
            if gi(&c["owner"]) == me {
                my_cities += 1;
            }
        }
    }
    // capture
    if let Some(a) = legal.iter().find(|a| kind(a) == 8) {
        return id(a);
    }
    // promote
    if let Some(a) = legal.iter().find(|a| kind(a) == 3 && par(a) == 16) {
        return id(a);
    }
    // attack: lowest health target
    let mut best: Option<(&Value, f64)> = None;
    for a in legal.iter().filter(|a| kind(a) == 2) {
        let t = tgt(a);
        if t < 0 || t as usize >= n {
            continue;
        }
        let s = src(a);
        let sh = if s >= 0 && (s as usize) < n { uhealth[s as usize] } else { 10.0 };
        let th = uhealth[t as usize];
        let mut sc = 100.0 - th + sh * 0.1;
        if structure[t as usize] == 1 {
            sc += 20.0;
        }
        if best.map_or(true, |b| sc > b.1) {
            best = Some((a, sc));
        }
    }
    if let Some((a, _)) = best {
        return id(a);
    }
    // targets
    let mut targets: Vec<(i64, i64)> = Vec::new(); // (tile, bonus)
    let mut unknowns: Vec<i64> = Vec::new();
    for i in 0..n {
        if !known[i] {
            unknowns.push(i as i64);
            continue;
        }
        if structure[i] == 1 && towner[i] != me {
            if uowner[i] == me {
                continue;
            }
            targets.push((i as i64, if towner[i] != 0 { 1 } else { 0 }));
        } else if structure[i] == 2 && uowner[i] != me {
            targets.push((i as i64, 1));
        } else if uowner[i] != 0 && uowner[i] != me {
            targets.push((i as i64, 3));
        }
    }
    let goal_cost = |t: i64| -> i64 {
        let mut b = 1000i64;
        for (g, bonus) in targets.iter() {
            let d = dist(t, *g) * 2 + bonus;
            if d < b {
                b = d;
            }
        }
        if b >= 1000 || b > 16 {
            let mut e = 1000i64;
            for g in unknowns.iter() {
                let d = dist(t, *g) * 2 + 6;
                if d < e {
                    e = d;
                }
            }
            if e < b {
                b = e;
            }
        }
        b
    };
    let stars_now = gi(&vs["stars"]);
    // moves
    let mut bestm: Option<(&Value, i64)> = None;
    for a in legal.iter().filter(|a| kind(a) == 1) {
        let s = src(a);
        let t = tgt(a);
        if s < 0 || t < 0 || s as usize >= n || t as usize >= n {
            continue;
        }
        // keep a unit sitting on a capturable tile
        let su = s as usize;
        if (structure[su] == 1 && towner[su] != me) || structure[su] == 2 {
            continue;
        }
        if structure[su] == 1 && towner[su] == me {
            let threat = (0..n).any(|i| uowner[i] != 0 && uowner[i] != me && dist(s, i as i64) <= 3);
            if threat || stars_now < 2 || my_units >= my_cities * 3 + 2 {
                continue;
            }
        }
        let gain = goal_cost(s) - goal_cost(t);
        if gain <= 0 {
            continue;
        }
        if bestm.map_or(true, |b| gain > b.1) {
            bestm = Some((a, gain));
        }
    }
    if let Some((a, _)) = bestm {
        return id(a);
    }
    let stars = gi(&vs["stars"]);
    // refill empty cities first
    {
        let pref = [11i64, 2, 3, 5, 9];
        for p in pref.iter() {
            if let Some(a) = legal.iter().find(|a| kind(a) == 4 && par(a) == *p) {
                let s0 = src(a);
                if s0 >= 0 && (s0 as usize) < n && structure[s0 as usize] == 1 && my_units < my_cities * 3 + 2 {
                    return id(a);
                }
            }
        }
    }
    // harvest
    if let Some(a) = legal.iter().find(|a| kind(a) == 5) {
        return id(a);
    }
    let summon = || -> Option<String> {
        if my_units >= my_cities * 3 + 2 {
            return None;
        }
        let pref = [11i64, 3, 2, 5, 9];
        for p in pref.iter() {
            if let Some(a) = legal.iter().find(|a| kind(a) == 4 && par(a) == *p) {
                return Some(id(a));
            }
        }
        None
    };
    let research = || -> Option<String> {
        let pref = [6i64, 1, 20, 10, 23, 8, 24, 13, 7, 18, 4, 2, 3, 44, 16, 9, 21, 17];
        for p in pref.iter() {
            if let Some(a) = legal.iter().find(|a| kind(a) == 7 && par(a) == *p) {
                return Some(id(a));
            }
        }
        None
    };
    let build = || -> Option<String> {
        let pref = [5i64, 21, 12, 8, 22, 6, 13];
        for p in pref.iter() {
            if let Some(a) = legal.iter().find(|a| kind(a) == 6 && par(a) == *p) {
                if *p == 8 && (0..n).any(|i| structure[i] == 8 && towner[i] == me) {
                    continue;
                }
                return Some(id(a));
            }
        }
        None
    };
    if my_units < my_cities * 2 + 1 {
        if let Some(x) = summon() {
            return x;
        }
    }
    if let Some(x) = build() {
        return x;
    }
    if my_units >= my_cities + 1 || stars >= 12 {
        if let Some(x) = research() {
            return x;
        }
    }
    if let Some(x) = summon() {
        return x;
    }
    fallback()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem { turn: -1, count: 0 };
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
        let a = choose(&obs, &mut mem);
        writeln!(stdout, "{}", json!({ "action_id": a })).unwrap();
        stdout.flush().unwrap();
    }
}
