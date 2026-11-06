use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Clone, Debug)]
struct Action {
    id: String,
    kind: i32,
    source: i32,
    target: i32,
    parameter: i32,
}

#[derive(Clone, Copy, Debug)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
}

#[derive(Clone, Debug)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    production: i32,
    progress: i32,
}

#[derive(Clone, Debug)]
struct Board {
    size: usize,
    n: usize,
    me: i32,
    turn: i32,
    stars: i32,
    tech: Vec<i32>,
    known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    road: Vec<bool>,
    capital: Vec<i32>,
    units: Vec<Unit>,
    cities: Vec<City>,
}

#[derive(Clone, Debug)]
struct Memory {
    size: usize,
    ever_known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    road: Vec<bool>,
    capital: Vec<i32>,
}

impl Memory {
    fn new(size: usize) -> Self {
        let n = size * size;
        Memory {
            size,
            ever_known: vec![false; n],
            terrain: vec![0; n],
            owner: vec![0; n],
            structure: vec![0; n],
            resource: vec![0; n],
            road: vec![false; n],
            capital: vec![0; n],
        }
    }
    fn update(&mut self, b: &Board) {
        for t in 0..b.n {
            if b.known[t] {
                self.ever_known[t] = true;
                self.terrain[t] = b.terrain[t];
                self.owner[t] = b.owner[t];
                self.structure[t] = b.structure[t];
                self.resource[t] = b.resource[t];
                self.road[t] = b.road[t];
                self.capital[t] = b.capital[t];
            }
        }
    }
}

fn get_i(v: &Value, key: &str) -> i32 {
    v.get(key).and_then(Value::as_i64).unwrap_or(0) as i32
}
fn get_f(v: &Value, key: &str) -> f32 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0) as f32
}
fn get_b(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn cmd(action: &Value) -> Action {
    let c = &action["command"];
    Action {
        id: action["action_id"].as_str().unwrap_or("").to_string(),
        kind: get_i(c, "kind"),
        source: get_i(c, "source"),
        target: get_i(c, "target"),
        parameter: get_i(c, "parameter"),
    }
}

fn parse_board(obs: &Value) -> Board {
    let vs = &obs["visible_state"];
    let size = get_i(vs, "board_size") as usize;
    let n = size * size;
    let me = get_i(obs, "player");
    let turn = get_i(vs, "turn");
    let stars = get_i(vs, "stars");
    let mut b = Board {
        size, n, me, turn, stars,
        tech: Vec::new(),
        known: vec![false; n],
        terrain: vec![0; n],
        owner: vec![0; n],
        structure: vec![0; n],
        resource: vec![0; n],
        road: vec![false; n],
        capital: vec![0; n],
        units: Vec::new(),
        cities: Vec::new(),
    };
    if let Some(arr) = vs.get("technology").and_then(Value::as_array) {
        for t in arr {
            if let Some(x) = t.as_i64() {
                b.tech.push(x as i32);
            }
        }
    }
    if let Some(tiles) = vs.get("tiles").and_then(Value::as_array) {
        for t in tiles {
            let idx = get_i(t, "tile") as usize;
            if idx >= n { continue; }
            let known = get_b(t, "known");
            b.known[idx] = known;
            if known {
                b.terrain[idx] = get_i(t, "terrain");
                b.owner[idx] = get_i(t, "owner");
                b.structure[idx] = get_i(t, "structure");
                b.resource[idx] = get_i(t, "resource");
                b.road[idx] = get_i(t, "road") != 0;
                b.capital[idx] = get_i(t, "capital");
            }
        }
    }
    if let Some(units) = vs.get("units").and_then(Value::as_array) {
        for u in units {
            let tile = get_i(u, "tile") as usize;
            if tile >= n { continue; }
            b.units.push(Unit {
                tile,
                owner: get_i(u, "owner"),
                kind: get_i(u, "kind"),
                health: get_f(u, "health"),
            });
        }
    }
    if let Some(cities) = vs.get("cities").and_then(Value::as_array) {
        for c in cities {
            let tile = get_i(c, "tile") as usize;
            if tile >= n { continue; }
            b.cities.push(City {
                tile,
                owner: get_i(c, "owner"),
                level: get_i(c, "level"),
                production: get_i(c, "production"),
                progress: get_i(c, "progress"),
            });
        }
    }
    b
}

fn dist_m(b: &Board, a: usize, c: usize) -> i32 {
    let s = b.size as i32;
    let (ax, ay) = ((a % b.size) as i32, (a / b.size) as i32);
    let (cx, cy) = ((c % b.size) as i32, (c / b.size) as i32);
    (ax - cx).abs().max((ay - cy).abs())
}
fn dist_s(size: usize, a: usize, c: usize) -> i32 {
    let s = size as i32;
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (cx, cy) = ((c % size) as i32, (c / size) as i32);
    (ax - cx).abs().max((ay - cy).abs())
}

fn neighbors(size: usize, t: usize) -> Vec<usize> {
    let s = size as i32;
    let x = (t % size) as i32;
    let y = (t / size) as i32;
    let mut out = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < s && ny >= 0 && ny < s {
                out.push((ny * s + nx) as usize);
            }
        }
    }
    out
}

fn unit_at<'a>(b: &'a Board, t: usize) -> Option<&'a Unit> {
    b.units.iter().find(|u| u.tile == t)
}
fn my_unit_at<'a>(b: &'a Board, t: usize) -> Option<&'a Unit> {
    b.units.iter().find(|u| u.tile == t && u.owner == b.me)
}
fn any_enemy_near(b: &Board, t: usize, radius: i32) -> bool {
    b.units.iter().any(|u| u.owner != b.me && u.owner != 0 && dist_m(b, u.tile, t) <= radius)
}

fn unit_stats(kind: i32) -> (f32, f32, f32, f32, i32, i32, bool) {
    match kind {
        2 => (2.0, 2.0, 10.0, 2.0, 1, 1, true),
        3 => (2.0, 1.0, 10.0, 3.0, 2, 1, true),
        4 => (3.5, 1.0, 10.0, 8.0, 3, 1, true),
        5 => (1.0, 3.0, 15.0, 3.0, 1, 1, false),
        8 => (4.0, 0.0, 10.0, 8.0, 1, 3, false),
        9 => (2.0, 1.0, 10.0, 3.0, 1, 2, true),
        10 => (0.0, 1.0, 10.0, 5.0, 1, 1, false),
        11 => (3.0, 3.0, 15.0, 5.0, 1, 1, true),
        12 => (5.0, 4.0, 40.0, 10.0, 1, 1, false),
        39 => (0.0, 0.5, 5.0, 2.0, 2, 1, true),
        40 => (2.0, 2.0, 10.0, 2.0, 2, 1, true),
        41 => (3.0, 2.0, 10.0, 15.0, 2, 3, false),
        42 => (2.0, 1.0, 10.0, 5.0, 3, 2, true),
        43 => (0.0, 1.0, 10.0, 0.0, 2, 2, false),
        44 => (3.0, 3.0, 10.0, 5.0, 3, 1, true),
        45 => (4.0, 4.0, 40.0, 10.0, 2, 1, false),
        _ => (1.0, 1.0, 10.0, 3.0, 1, 1, true),
    }
}

fn defense_bonus_for_terrain(b: &Board, t: usize, defender_owner: i32) -> f32 {
    let terrain = b.terrain[t];
    let city = b.structure[t] == 1 && b.owner[t] == defender_owner;
    if terrain == 5 || terrain == 4 || terrain == 1 || terrain == 2 {
        if city { 2.0 } else { 1.5 }
    } else if city {
        1.5
    } else {
        1.0
    }
}

fn combat_damage(att: (f32, f32, f32), def: (f32, f32, f32), att_hp: f32, def_hp: f32, def_bonus: f32) -> (f32, f32) {
    let (aa, _ad, am) = att;
    let (da, dd, dm) = def;
    let af = aa * (att_hp / am.max(1.0));
    let df = dd * (def_hp / dm.max(1.0)) * def_bonus;
    let total = af + df;
    if total <= 0.0 { return (0.0, 0.0); }
    let dmg = ((af / total) * aa * 4.5 + 0.000001 + 0.5).floor();
    let ret = if def_hp > dmg { ((df / total) * da * 4.5 + 0.000001 + 0.5).floor() } else { 0.0 };
    (dmg, ret)
}

fn nearest_tile<F: Fn(usize) -> bool>(size: usize, n: usize, from: usize, f: F) -> Option<(usize, i32)> {
    let mut best = None;
    for t in 0..n {
        if f(t) {
            let d = dist_s(size, from, t);
            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((t, d));
            }
        }
    }
    best
}

fn water_ratio(b: &Board) -> f32 {
    let mut known = 0usize;
    let mut water = 0usize;
    for t in 0..b.n {
        if b.known[t] {
            known += 1;
            if b.terrain[t] == 1 || b.terrain[t] == 2 { water += 1; }
        }
    }
    if known == 0 { 0.0 } else { water as f32 / known as f32 }
}

fn count_terrain(b: &Board, terrain: i32) -> i32 {
    (0..b.n).filter(|&t| b.known[t] && b.terrain[t] == terrain).count() as i32
}

fn my_city_count(b: &Board) -> usize {
    b.cities.iter().filter(|c| c.owner == b.me).count()
}
fn my_unit_count(b: &Board) -> usize {
    b.units.iter().filter(|u| u.owner == b.me).count()
}

fn tech_score(b: &Board, tech: i32) -> i32 {
    let water = water_ratio(b);
    let fruit = (0..b.n).any(|t| b.known[t] && b.resource[t] == 6 && b.owner[t] == b.me) as i32;
    let crop = (0..b.n).any(|t| b.known[t] && b.resource[t] == 2 && b.owner[t] == b.me) as i32;
    let fish = (0..b.n).any(|t| b.known[t] && b.resource[t] == 3 && b.owner[t] == b.me) as i32;
    let forest = count_terrain(b, 5);
    let mountain = count_terrain(b, 4);
    let near_enemy = b.units.iter().any(|u| u.owner != b.me && u.owner != 0) as i32;
    let mut s = match tech {
        1 => 125, // Riding -> riders
        6 => 90 + fruit * 30 + crop * 10, // Organization
        10 => if water > 0.30 || fish > 0 { 135 } else { 30 }, // Fishing
        8 => 75 + crop * 30 + fruit * 10,
        4 => 90,
        18 => 72 + if forest > 4 { 20 } else { 0 },
        16 => 65 + if forest > 4 { 25 } else { 0 },
        23 => 55 + mountain * 10,
        7 => 58 + near_enemy * 25,
        13 => if water > 0.2 { 110 } else { 10 },
        14 => if water > 0.2 { 95 } else { 5 },
        44 => if water > 0.2 { 75 } else { 5 },
        12 => if water > 0.2 { 65 } else { 5 },
        17 => 68 + if forest > 4 { 20 } else { 0 },
        24 => 60 + mountain * 10,
        3 => 60,
        5 => 50,
        9 => 50,
        2 => 25,
        19 => 30,
        20 => 35,
        21 => 25,
        22 => 20,
        38 => 10,
        _ => 0,
    };
    // Keep enough stars for a warrior/rider unless we are rich.
    if b.stars <= 5 && my_unit_count(b) < 2 {
        s -= 35;
    }
    s
}

fn reward_score(_b: &Board, param: i32) -> i32 {
    match param {
        6 => 1000,
        3 => 800,
        7 => 750,
        5 => 700,
        8 => 650,
        1 => 550,
        2 => 500,
        4 => 400,
        _ => 0,
    }
}

fn harvest_score(b: &Board, target: usize) -> i32 {
    let r = b.resource[target];
    let mut base = match r {
        1 | 6 => 120,
        3 => 110,
        _ => 80,
    };
    let city_tile = nearest_own_city_for_tile(b, target);
    if let Some(city) = city_tile {
        if let Some(c) = b.cities.iter().find(|c| c.tile == city) {
            let need = c.level + 1 - c.progress;
            if need <= 1 { base += 80; }
            else if need <= 2 { base += 40; }
        }
    }
    base
}

fn nearest_own_city_for_tile(b: &Board, t: usize) -> Option<usize> {
    let mut best = None;
    for c in &b.cities {
        if c.owner == b.me {
            let d = dist_m(b, c.tile, t);
            if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((c.tile, d));
            }
        }
    }
    best.map(|x| x.0)
}

fn structure_score(b: &Board, param: i32, target: usize) -> i32 {
    let r = b.resource[target];
    let terrain = b.terrain[target];
    let water = terrain == 1 || terrain == 2;
    match param {
        5 => if r == 2 { 150 } else { 60 },
        21 => if r == 5 { 150 } else { 60 },
        12 => if terrain == 5 { 100 } else { 40 },
        8 => if water && water_ratio(b) > 0.15 { 120 } else { 30 },
        3 => 30,
        48 => 40,
        6 => 80,
        13 => 85,
        22 => 80,
        50 => 70,
        17 => 50,
        18 => 45,
        19 => 45,
        20 => 45,
        _ => 20,
    }
}

fn summon_score(b: &Board, source: usize, param: i32) -> i32 {
    let near_enemy = b.units.iter().any(|u| u.owner != b.me && u.owner != 0);
    let mut s = match param {
        12 => 180,
        4 => 120,
        3 => 125,
        11 => 115,
        9 => 105,
        42 => 110,
        44 => 105,
        41 => 110,
        5 => if near_enemy { 110 } else { 70 },
        2 => 88,
        43 => 50,
        10 => 40,
        8 => 95,
        _ => 30,
    };
    // Riders are the best general unit once available.
    if param == 3 && b.tech.contains(&1) { s += 15; }
    // Recruit from the capital first, especially when it is empty.
    if b.capital.get(source).copied().unwrap_or(0) == b.me { s += 250; }
    if b.structure.get(source).copied().unwrap_or(0) == 1 && b.owner.get(source).copied().unwrap_or(0) == b.me && b.capital.get(source).copied().unwrap_or(0) == b.me { s += 100; }
    s
}

fn passable_for_unit(b: &Board, mem: &Memory, t: usize, kind: i32) -> bool {
    if !mem.ever_known[t] { return false; }
    let terrain = mem.terrain[t];
    // Ships and amphibious units use water; ordinary land units need land or a port.
    let ship = matches!(kind, 39 | 40 | 41 | 42 | 43 | 44 | 45);
    if terrain == 1 || terrain == 2 {
        return ship || mem.structure[t] == 8;
    }
    if terrain == 4 {
        return b.tech.contains(&20); // Climbing
    }
    true
}

fn goal_for_unit(b: &Board, mem: &Memory, src: usize) -> usize {
    let me = b.me;
    let kind = my_unit_at(b, src).map(|u| u.kind).unwrap_or(2);
    // Remembered neutral villages (unowned, structure 1).
    if let Some((t, _)) = nearest_tile(mem.size, b.n, src, |t| {
        mem.ever_known[t] && mem.structure[t] == 1 && mem.owner[t] == 0 && passable_for_unit(b, mem, t, kind)
    }) {
        return t;
    }
    // Ruins.
    if let Some((t, _)) = nearest_tile(mem.size, b.n, src, |t| {
        mem.ever_known[t] && mem.structure[t] == 2 && passable_for_unit(b, mem, t, kind)
    }) {
        return t;
    }
    // If we have several units, move toward a remembered enemy city.
    if my_unit_count(b) >= 3 {
        if let Some((t, _)) = nearest_tile(mem.size, b.n, src, |t| {
            mem.ever_known[t] && mem.structure[t] == 1 && mem.owner[t] != 0 && mem.owner[t] != me && passable_for_unit(b, mem, t, kind)
        }) {
            return t;
        }
    }
    // A stable frontier tile of the union of everything we have ever seen.
    if let Some((t, _)) = nearest_tile(mem.size, b.n, src, |t| {
        mem.ever_known[t] && passable_for_unit(b, mem, t, kind) && neighbors(mem.size, t).iter().any(|&n| !mem.ever_known[n])
    }) {
        return t;
    }
    b.n / 2
}

fn assign_goals(b: &Board, mem: &Memory) -> Vec<(usize, usize)> {
    let mut used: Vec<usize> = Vec::new();
    let mut goals = Vec::new();
    let me = b.me;
    let mut own: Vec<usize> = b.units.iter().filter(|u| u.owner == me).map(|u| u.tile).collect();
    own.sort();
    for src in own {
        let kind = my_unit_at(b, src).map(|u| u.kind).unwrap_or(2);
        let mut candidates: Vec<(usize, i32)> = (0..b.n)
            .filter(|&t| {
                mem.ever_known[t] && passable_for_unit(b, mem, t, kind)
                    && ((mem.structure[t] == 1 && mem.owner[t] == 0)
                        || mem.structure[t] == 2
                        || neighbors(mem.size, t).iter().any(|&n| !mem.ever_known[n]))
            })
            .map(|t| (t, dist_m(b, src, t)))
            .collect();
        candidates.sort_by_key(|&(t, d)| (d, t));
        let mut chosen = None;
        // Prefer a target not already claimed and not too close to another claimed one.
        for &(t, d) in &candidates {
            if used.contains(&t) { continue; }
            if used.iter().any(|&u| dist_s(mem.size, t, u) <= 1) { continue; }
            chosen = Some((t, d));
            break;
        }
        if chosen.is_none() {
            for &(t, _) in &candidates {
                if !used.contains(&t) { chosen = Some((t, dist_m(b, src, t))); break; }
            }
        }
        let goal = chosen.map(|(t, _)| t).unwrap_or_else(|| goal_for_unit(b, mem, src));
        used.push(goal);
        goals.push((src, goal));
    }
    goals
}

fn best_action(obs: &Value, mem: &mut Memory) -> String {
    let legal = obs["legal_actions"].as_array().expect("legal_actions");
    let actions: Vec<Action> = legal.iter().map(cmd).collect();
    if actions.is_empty() {
        return String::new();
    }
    let b = parse_board(obs);
    mem.update(&b);

    // 1. Mandatory city reward.
    let mut rewards: Vec<&Action> = actions.iter().filter(|a| a.kind == 9).collect();
    rewards.sort_by_key(|a| -reward_score(&b, a.parameter));
    if let Some(a) = rewards.first() {
        return a.id.clone();
    }

    // 2. Promote.
    if let Some(a) = actions.iter().find(|a| a.kind == 3 && a.parameter == 16) {
        return a.id.clone();
    }

    // 3. Capture.
    if let Some(a) = actions.iter().find(|a| a.kind == 8) {
        return a.id.clone();
    }

    // 4. Attack favorable targets.
    let mut best_attack: Option<(i32, &Action)> = None;
    for a in actions.iter().filter(|a| a.kind == 2) {
        let src = a.source as usize;
        let dst = a.target as usize;
        if let Some(au) = my_unit_at(&b, src) {
            if let Some(du) = unit_at(&b, dst) {
                let att = unit_stats(au.kind);
                let def = unit_stats(du.kind);
                let bonus = defense_bonus_for_terrain(&b, dst, du.owner);
                let (dmg, ret) = combat_damage((att.0, att.1, att.2), (def.0, def.1, def.2), au.health, du.health, bonus);
                let kill = dmg >= du.health;
                let our_death = ret >= au.health;
                let enemy_cost = def.3;
                let our_cost = att.3;
                let mut s = dmg as i32 * 10;
                if kill { s += 150 + (enemy_cost * 5.0) as i32; }
                s -= (ret * 8.0) as i32;
                if our_death { s -= 200 + (our_cost * 5.0) as i32; }
                if kill && !our_death { s += 100; }
                if our_death && !kill { s -= 500; }
                if best_attack.map(|(bs, _)| s > bs).unwrap_or(true) {
                    best_attack = Some((s, a));
                }
            }
        }
    }
    if let Some((s, a)) = best_attack {
        if s > 35 {
            return a.id.clone();
        }
    }

    // 5. Recover injured units away from the front.
    if let Some(a) = actions.iter().find(|a| a.kind == 3 && a.parameter == 7) {
        let src = a.source as usize;
        if let Some(u) = my_unit_at(&b, src) {
            let maxhp = unit_stats(u.kind).2;
            if u.health < maxhp * 0.7 && !any_enemy_near(&b, src, 2) {
                return a.id.clone();
            }
        }
    }

    // 6. Research. Early: one cheap tech before expanding. Later: only when no tactical action and we are rich.
    let mut research: Vec<&Action> = actions.iter().filter(|a| a.kind == 7).collect();
    research.sort_by_key(|a| -tech_score(&b, a.parameter));
    if let Some(a) = research.first() {
        let sc = tech_score(&b, a.parameter);
        let early = b.turn <= 6 && my_city_count(&b) <= 2 && my_unit_count(&b) <= 1;
        let rich = b.stars >= 10 && sc > 70;
        let no_other = !actions.iter().any(|a| a.kind == 8 || a.kind == 2 || a.kind == 9);
        if (early && b.stars >= 4 && sc > 70 && !actions.iter().any(|a| a.kind == 8)) || (rich && no_other) {
            return a.id.clone();
        }
    }

    // 7. Move/explore.
    let assigned = assign_goals(&b, mem);
    let mut best_move: Option<(i32, &Action)> = None;
    for a in actions.iter().filter(|a| a.kind == 1) {
        let src = a.source as usize;
        let dst = a.target as usize;
        if src >= b.n || dst >= b.n { continue; }
        let goal = assigned.iter().find(|(s, _)| *s == src).map(|&(_, g)| g).unwrap_or_else(|| goal_for_unit(&b, mem, src));
        let current_d = dist_m(&b, src, goal);
        let new_d = dist_m(&b, dst, goal);
        let mut s = 45 + (current_d - new_d) * 22;
        if neighbors(mem.size, dst).iter().any(|&n| !mem.ever_known[n]) { s += 35; }
        if mem.structure[dst] == 2 { s += 100; }
        if mem.structure[dst] == 1 && mem.owner[dst] == 0 { s += 80; }
        if mem.structure[dst] == 1 && mem.owner[dst] != b.me && mem.owner[dst] != 0 { s += 120; }
        if any_enemy_near(&b, dst, 1) { s -= 45; }
        // Keep a garrison in each own city when the army is no longer a lone explorer.
        let src_is_own_city = b.structure.get(src).copied().unwrap_or(0) == 1 && b.owner.get(src).copied().unwrap_or(0) == b.me;
        if src_is_own_city && my_unit_count(&b) > 1 {
            let city_has_other_defender = b.units.iter().any(|u| u.owner == b.me && u.tile != src && dist_m(&b, u.tile, src) <= 1);
            if !city_has_other_defender {
                s -= 300;
            }
        }
        if best_move.map(|(bs, _)| s > bs).unwrap_or(true) {
            best_move = Some((s, a));
        }
    }
    if let Some((s, a)) = best_move {
        if s > 35 {
            return a.id.clone();
        }
    }

    // 8. Recruit units.
    let mut summons: Vec<&Action> = actions.iter().filter(|a| a.kind == 4).collect();
    summons.sort_by_key(|a| -summon_score(&b, a.source as usize, a.parameter));
    if let Some(a) = summons.first() {
        let cost = unit_stats(a.parameter).3 as i32;
        let sc = summon_score(&b, a.source as usize, a.parameter);
        let capacity: i32 = b.cities.iter().filter(|c| c.owner == b.me).map(|c| c.level).sum();
        let can_room = my_unit_count(&b) < capacity as usize + 2;
        if b.stars >= cost && sc >= 65 && can_room {
            return a.id.clone();
        }
    }

    // 9. Harvest.
    if b.stars >= 4 {
        let mut harvests: Vec<&Action> = actions.iter().filter(|a| a.kind == 5).collect();
        harvests.sort_by_key(|a| -harvest_score(&b, a.target as usize));
        if let Some(a) = harvests.first() {
            if harvest_score(&b, a.target as usize) > 100 {
                return a.id.clone();
            }
        }
    }

    // 10. Build improvements.
    let mut builds: Vec<&Action> = actions.iter().filter(|a| a.kind == 6).collect();
    builds.sort_by_key(|a| -structure_score(&b, a.parameter, a.target as usize));
    if let Some(a) = builds.first() {
        let sc = structure_score(&b, a.parameter, a.target as usize);
        let cost = match a.parameter {
            3 => 3, 5 => 5, 6 => 5, 8 => 7, 12 => 3, 13 => 5, 21 => 5, 22 => 5, 50 => 5, 48 => 5, _ => 5,
        };
        let reserve = if b.stars < 10 { 3 } else { 1 };
        if sc >= 100 && b.stars >= cost + reserve {
            return a.id.clone();
        }
        if a.parameter == 3 && b.stars >= 8 && my_city_count(&b) >= 2 {
            return a.id.clone();
        }
    }

    // 11. End turn.
    if let Some(a) = actions.iter().find(|a| a.kind == 10) {
        return a.id.clone();
    }

    // 12. Fallback (anything except resign).
    if let Some(a) = actions.iter().find(|a| a.kind != 11) {
        return a.id.clone();
    }
    actions[0].id.clone()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem: Option<Memory> = None;
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let obs: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let size = get_i(&obs["visible_state"], "board_size") as usize;
        if mem.is_none() || mem.as_ref().map(|m| m.size != size).unwrap_or(false) {
            mem = Some(Memory::new(size));
        }
        let id = best_action(&obs, mem.as_mut().unwrap());
        if id.is_empty() { continue; }
        writeln!(stdout, "{}", json!({"action_id": id})).unwrap();
        stdout.flush().unwrap();
    }
}
