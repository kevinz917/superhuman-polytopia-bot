// Polytopia Bardur 1v1 bot: heuristic policy over the public observation.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

#[derive(Clone, Copy, Debug)]
struct US {
    cost: i32,
    hp: f32,
    atk: f32,
    def: f32,
    mov: i32,
    rng: i32,
    stat: bool,
    stiff: bool,
}

fn stat(kind: i64) -> Option<US> {
    Some(match kind {
        2 => US { cost: 2, hp: 10.0, atk: 2.0, def: 2.0, mov: 1, rng: 1, stat: false, stiff: false },
        3 => US { cost: 3, hp: 10.0, atk: 2.0, def: 1.0, mov: 2, rng: 1, stat: false, stiff: false },
        4 => US { cost: 8, hp: 10.0, atk: 3.5, def: 1.0, mov: 3, rng: 1, stat: false, stiff: false },
        5 => US { cost: 3, hp: 15.0, atk: 1.0, def: 3.0, mov: 1, rng: 1, stat: false, stiff: false },
        8 => US { cost: 8, hp: 10.0, atk: 4.0, def: 0.0, mov: 1, rng: 3, stat: false, stiff: true },
        9 => US { cost: 3, hp: 10.0, atk: 2.0, def: 1.0, mov: 1, rng: 2, stat: false, stiff: false },
        11 => US { cost: 5, hp: 15.0, atk: 3.0, def: 3.0, mov: 1, rng: 1, stat: false, stiff: false },
        12 => US { cost: 10, hp: 40.0, atk: 5.0, def: 4.0, mov: 1, rng: 1, stat: true, stiff: false },
        43 => US { cost: 0, hp: 10.0, atk: 0.0, def: 1.0, mov: 2, rng: 2, stat: true, stiff: true },
        44 => US { cost: 5, hp: 10.0, atk: 3.0, def: 3.0, mov: 3, rng: 1, stat: true, stiff: false },
        _ => return None,
    })
}

#[derive(Clone)]
struct Tile {
    known: bool,
    terrain: i64,
    owner: i64,
    structure: i64,
    resource: i64,
    road: bool,
    capital: i64,
}

#[derive(Clone)]
struct U {
    tile: usize,
    owner: i64,
    kind: i64,
    hp: f32,
    vet: bool,
    moved: bool,
    attacked: bool,
    kills: i64,
    home: i64,
    passenger: i64,
    invis: bool,
}

#[derive(Clone)]
struct City {
    tile: usize,
    owner: i64,
    level: i64,
    prod: i64,
    prog: i64,
    pop: i64,
    connected: bool,
    rewards: Vec<i64>,
}

#[derive(Clone)]
struct Act {
    id: String,
    kind: i64,
    source: i64,
    target: i64,
    param: i64,
}

struct St {
    size: usize,
    me: i64,
    turn: i64,
    stars: i64,
    tech: Vec<bool>,
    tiles: Vec<Tile>,
    units: Vec<U>,
    cities: Vec<City>,
    at: Vec<i64>,
}

fn gi(v: &Value, k: &str) -> i64 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0)
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}
fn gf(v: &Value, k: &str) -> f32 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
}

fn parse(obs: &Value) -> St {
    let v = &obs["visible_state"];
    let size = gi(v, "board_size").max(2) as usize;
    let n = size * size;
    let me = gi(obs, "player");
    let turn = gi(v, "turn");
    let stars = gi(v, "stars");
    let mut tech = vec![false; 128];
    if let Some(t) = v["technology"].as_array() {
        for x in t {
            if let Some(k) = x.as_i64() {
                if (0..128).contains(&k) {
                    tech[k as usize] = true;
                }
            }
        }
    }
    let mut tiles = vec![Tile { known: false, terrain: 0, owner: 0, structure: 0, resource: 0, road: false, capital: 0 }; n];
    if let Some(a) = v["tiles"].as_array() {
        for x in a {
            let i = gi(x, "tile");
            if i < 0 || i as usize >= n {
                continue;
            }
            let i = i as usize;
            tiles[i] = Tile {
                known: gb(x, "known"),
                terrain: gi(x, "terrain"),
                owner: gi(x, "owner"),
                structure: gi(x, "structure"),
                resource: gi(x, "resource"),
                road: gi(x, "road") != 0,
                capital: gi(x, "capital"),
            };
        }
    }
    let mut units: Vec<U> = Vec::new();
    if let Some(a) = v["units"].as_array() {
        for x in a {
            let t = gi(x, "tile");
            if t < 0 || t as usize >= n {
                continue;
            }
            units.push(U {
                tile: t as usize,
                owner: gi(x, "owner"),
                kind: gi(x, "kind"),
                hp: gf(x, "health"),
                vet: gb(x, "veteran"),
                moved: gb(x, "moved"),
                attacked: gb(x, "attacked"),
                kills: gi(x, "kills"),
                home: gi(x, "home"),
                passenger: gi(x, "passenger"),
                invis: gb(x, "invisible"),
            });
        }
    }
    let mut cities: Vec<City> = Vec::new();
    if let Some(a) = v["cities"].as_array() {
        for x in a {
            let t = gi(x, "tile");
            if t < 0 || t as usize >= n {
                continue;
            }
            let mut rw = vec![0i64; 9];
            if let Some(r) = x["rewards"].as_array() {
                for (i, z) in r.iter().enumerate().take(9) {
                    rw[i] = z.as_i64().unwrap_or(0);
                }
            }
            cities.push(City {
                tile: t as usize,
                owner: gi(x, "owner"),
                level: gi(x, "level"),
                prod: gi(x, "production"),
                prog: gi(x, "progress"),
                pop: gi(x, "population"),
                connected: gb(x, "connected"),
                rewards: rw,
            });
        }
    }
    let mut at = vec![-1i64; n];
    for (i, u) in units.iter().enumerate() {
        if at[u.tile] < 0 {
            at[u.tile] = i as i64;
        }
    }
    St { size, me, turn, stars, tech, tiles, units, cities, at }
}

fn parse_acts(obs: &Value) -> Vec<Act> {
    let mut out = Vec::new();
    if let Some(a) = obs["legal_actions"].as_array() {
        for x in a {
            let c = &x["command"];
            out.push(Act {
                id: x["action_id"].as_str().unwrap_or("").to_string(),
                kind: gi(c, "kind"),
                source: gi(c, "source"),
                target: gi(c, "target"),
                param: gi(c, "parameter"),
            });
        }
    }
    out
}

fn cheb(size: usize, a: usize, b: usize) -> i32 {
    let ax = (a % size) as i32;
    let ay = (a / size) as i32;
    let bx = (b % size) as i32;
    let by = (b / size) as i32;
    (ax - bx).abs().max((ay - by).abs())
}

struct Ctx {
    my_cities: Vec<usize>,
    enemy_cities: Vec<usize>,
    enemy_cap: Option<usize>,
    my_units: Vec<usize>,
    enemy_units: Vec<usize>,
    villages: Vec<usize>,
    unknown: Vec<usize>,
    goals: Vec<usize>,
    nports: usize,
}

fn nearest(from: usize, targets: &[usize], size: usize) -> Option<usize> {
    let mut best: Option<(i32, usize)> = None;
    for &t in targets {
        let d = cheb(size, from, t);
        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, t));
        }
    }
    best.map(|(_, t)| t)
}

fn build_ctx(st: &St) -> Ctx {
    let size = st.size;
    let mut my_cities = Vec::new();
    let mut enemy_cities = Vec::new();
    let mut enemy_cap = None;
    for c in &st.cities {
        if c.owner == st.me {
            my_cities.push(c.tile);
        } else if c.owner > 0 {
            enemy_cities.push(c.tile);
            if st.tiles[c.tile].capital == c.owner {
                enemy_cap = Some(c.tile);
            }
        }
    }
    let mut my_units = Vec::new();
    let mut enemy_units = Vec::new();
    for (i, u) in st.units.iter().enumerate() {
        if u.owner == st.me {
            my_units.push(i);
        } else if u.owner > 0 {
            enemy_units.push(i);
        }
    }
    let mut villages = Vec::new();
    let mut unknown = Vec::new();
    let mut nports = 0usize;
    for (i, t) in st.tiles.iter().enumerate() {
        if !t.known {
            unknown.push(i);
            continue;
        }
        if t.structure == 1 && t.owner == 0 {
            villages.push(i);
        }
        if t.structure == 8 {
            nports += 1;
        }
    }
    let army = my_units.len();
    let attack_mode = (army >= 10 && my_cities.len() >= 3) || (army >= 14);
    // a threat: an enemy unit close to one of our cities
    let mut threat: Option<(i32, usize)> = None;
    for &ei in &enemy_units {
        let et = st.units[ei].tile;
        for &mc in &my_cities {
            let d = cheb(size, et, mc);
            if d <= 3 && threat.map(|(td, _)| d < td).unwrap_or(true) {
                threat = Some((d, et));
            }
        }
    }
    let mut goals = vec![usize::MAX; st.units.len()];
    for &ui in &my_units {
        let u = &st.units[ui];
        let hp_frac = u.hp / max_hp(st, ui).max(1.0);
        let mut goal = usize::MAX;
        let mut fight: Option<(i32, usize)> = None;
        for &ei in &enemy_units {
            let et = st.units[ei].tile;
            let d = cheb(size, u.tile, et);
            if d > 2 {
                continue;
            }
            let our_local = my_units.iter().filter(|&&x| cheb(size, st.units[x].tile, et) <= 1).count();
            let their_local = enemy_units.iter().filter(|&&x| cheb(size, st.units[x].tile, et) <= 1).count();
            if our_local > their_local && fight.map(|(fd, _)| d < fd).unwrap_or(true) {
                fight = Some((d, et));
            }
        }
        if let Some((_, t)) = fight {
            goal = t;
        } else if let Some((_, t)) = threat {
            // only nearby units defend
            let near_city = my_cities.iter().any(|&c| cheb(size, u.tile, c) <= 3);
            if near_city {
                goal = t;
            }
        }
        if goal == usize::MAX {
            // retreat when badly hurt and an enemy is close
            let hurt = hp_frac < 0.5;
            let enemy_close = enemy_units.iter().any(|&ei| cheb(size, u.tile, st.units[ei].tile) <= 3);
            if hurt && enemy_close {
                if let Some(c) = nearest(u.tile, &my_cities, size) {
                    goal = c;
                }
            }
        }
        if goal == usize::MAX && attack_mode {
            if let Some(c) = enemy_cap {
                goal = c;
            } else if let Some(c) = nearest(u.tile, &enemy_cities, size) {
                goal = c;
            }
        }
        if goal == usize::MAX {
            // expansion: nearest village, preferring villages close to our own territory
            let mut best: Option<(f32, usize)> = None;
            for &v in &villages {
                let dv = cheb(size, u.tile, v) as f32;
                let dc = my_cities.iter().map(|&c| cheb(size, c, v)).min().unwrap_or(0) as f32;
                let cost = dv + 0.7 * dc;
                if best.map(|(b, _)| cost < b).unwrap_or(true) {
                    best = Some((cost, v));
                }
            }
            if let Some((_, v)) = best {
                goal = v;
            }
        }
        if goal == usize::MAX {
            if let Some(v) = nearest(u.tile, &unknown, size) {
                goal = v;
            }
        }
        if goal == usize::MAX {
            goal = u.tile;
        }
        goals[ui] = goal;
    }
    Ctx { my_cities, enemy_cities, enemy_cap, my_units, enemy_units, villages, unknown, goals, nports }
}

// tech priorities: (id, tier, priority)
const TECHS: &[(i64, i64, f32)] = &[
    (6, 1, 720.0),   // Organization: farming + defender
    (1, 1, 712.0),   // Riding: rider + roads
    (10, 1, 706.0),  // Fishing: ports, boats
    (20, 1, 700.0),  // Climbing: mining
    (8, 2, 676.0),   // Farming: farms
    (18, 2, 670.0),  // Archery: archers
    (16, 2, 664.0),  // Forestry: lumber hut
    (7, 2, 658.0),   // Strategy: defender
    (4, 2, 652.0),   // Roads
    (2, 2, 646.0),   // FreeSpirit
    (9, 3, 630.0),   // Construction: windmill, clear forest
    (3, 3, 624.0),   // Chivalry: knight
    (17, 3, 618.0),  // Mathematics: catapult
    (24, 3, 612.0),  // Smithery: swordsman
    (21, 3, 606.0),  // Meditation
    (19, 3, 600.0),  // Spiritualism
    (22, 3, 594.0),  // Philosophy
];

fn tech_priority(id: i64) -> Option<(i64, f32)> {
    TECHS.iter().find(|t| t.0 == id).map(|t| (t.1, t.2))
}

fn unit_pref(kind: i64) -> f32 {
    match kind {
        4 => 120.0,  // knight
        11 => 110.0, // swordsman
        9 => 100.0,  // archer
        8 => 95.0,   // catapult
        3 => 80.0,   // rider
        2 => 70.0,   // warrior
        5 => 50.0,   // defender
        44 => 40.0,  // rammership
        43 => 30.0,  // raft
        12 => 40.0,  // giant
        _ => 10.0,
    }
}

fn max_hp(st: &St, i: usize) -> f32 {
    let u = &st.units[i];
    let k = if u.passenger > 0 { u.passenger } else { u.kind };
    let base = stat(k).map(|s| s.hp).unwrap_or(10.0);
    base + if u.vet { 5.0 } else { 0.0 }
}

fn score(st: &St, ctx: &Ctx, a: &Act) -> f32 {
    match a.kind {
        11 => -1.0e9,
        10 => 0.0,
        9 => reward_score(st, ctx, a),
        8 => capture_score(st, a),
        7 => research_score(st, a),
        4 => summon_score(st, a),
        2 => attack_score(st, ctx, a),
        5 => harvest_score(st, a),
        6 => build_score(st, ctx, a),
        3 => {
            if a.source >= 0 {
                match a.param {
                    16 => 820.0, // promote
                    7 => 320.0,  // recover
                    9 => 260.0,  // heal others
                    8 => 20.0,   // disband
                    _ => 0.0,
                }
            } else {
                match a.param {
                    1 => 250.0, // clear forest -> field
                    2 => 40.0,  // grow forest
                    3 => 210.0, // burn forest
                    4 => 30.0,  // destroy improvement
                    _ => 0.0,
                }
            }
        }
        1 => move_score(st, ctx, a),
        _ => 0.0,
    }
}

fn reward_score(st: &St, ctx: &Ctx, a: &Act) -> f32 {
    let _ = ctx;
    match a.param {
        3 => 520.0,                                    // workshop: +1 production
        4 => {                                         // explorer
            if ctx.unknown.len() * 3 > st.tiles.len() {
                545.0
            } else {
                495.0
            }
        }
        1 => {                                         // city wall
            if st.stars < 12 {
                505.0
            } else {
                535.0
            }
        }
        7 => {                                         // +5 stars
            if st.stars < 25 {
                545.0
            } else {
                500.0
            }
        }
        5 => 530.0,                                    // border growth
        8 => 550.0,                                    // population
        2 => 520.0,                                    // park
        6 => {                                         // giant
            if ctx.my_units.len() < 8 {
                515.0
            } else {
                470.0
            }
        }
        _ => 0.0,
    }
}

fn capture_score(st: &St, a: &Act) -> f32 {
    if a.source < 0 {
        return 0.0;
    }
    let t = &st.tiles[a.source as usize];
    if t.structure == 1 {
        if t.owner == 0 {
            900.0
        } else if t.owner != st.me {
            1400.0
        } else {
            0.0
        }
    } else if t.structure == 2 {
        430.0
    } else if t.resource == 8 {
        520.0
    } else {
        0.0
    }
}

fn research_score(st: &St, a: &Act) -> f32 {
    match tech_priority(a.param) {
        Some((tier, prio)) => {
            let cities = st.cities.iter().filter(|c| c.owner == st.me).count() as i64;
            let mut cost = 4 + cities * tier;
            if st.tech[22] {
                cost = ((cost as f32) * 0.66666).ceil() as i64;
            }
            let mut s = prio;
            if st.stars < cost + 3 {
                s -= 350.0;
            }
            s
        }
        None => 0.0,
    }
}

fn summon_score(_st: &St, a: &Act) -> f32 {
    let k = a.param;
    let cost = stat(k).map(|s| s.cost).unwrap_or(3);
    380.0 + unit_pref(k) - 2.0 * cost as f32
}

fn harvest_score(st: &St, _a: &Act) -> f32 {
    if st.stars >= 12 {
        210.0
    } else {
        120.0
    }
}

fn build_score(_st: &St, ctx: &Ctx, a: &Act) -> f32 {
    match a.param {
        5 => 370.0,                                     // farm
        6 => 330.0,                                     // windmill
        8 => {                                          // port
            if ctx.nports == 0 {
                360.0
            } else {
                120.0
            }
        }
        13 => 195.0,                                    // sawmill
        17 => 235.0,                                    // temple
        12 => 175.0,                                    // lumber hut
        21 => 175.0,                                    // mine
        22 => 175.0,                                    // forge
        3 => 150.0,                                     // road
        48 => 140.0,                                    // bridge
        23..=29 => 70.0,                                // monuments
        _ => 40.0,
    }
}

fn attack_score(st: &St, ctx: &Ctx, a: &Act) -> f32 {
    if a.source < 0 || a.target < 0 {
        return 0.0;
    }
    let si = st.at[a.source as usize];
    if si < 0 {
        return 0.0;
    }
    let si = si as usize;
    if st.units[si].owner != st.me {
        return 0.0;
    }
    let ti = st.at[a.target as usize];
    if ti < 0 {
        return 0.0;
    }
    let ti = ti as usize;
    let du = &st.units[ti];
    if du.owner == st.me || du.owner == 0 {
        return 0.0;
    }
    let ds = match stat(du.kind) {
        Some(s) => s,
        None => return 0.0,
    };
    let us_ = match stat(st.units[si].kind) {
        Some(s) => s,
        None => return 0.0,
    };
    let our_max = max_hp(st, si);
    let def_max = ds.hp + if du.vet { 5.0 } else { 0.0 };
    let hp_frac = st.units[si].hp / our_max.max(1.0);
    let af = us_.atk * (st.units[si].hp / our_max.max(1.0));
    let terr = st.tiles[a.target as usize].terrain;
    let tt = &st.tiles[a.target as usize];
    let on_enemy_city = tt.structure == 1 && tt.owner != st.me;
    let on_our_city = tt.structure == 1 && tt.owner == st.me;
    let mut bonus = 1.0f32;
    if terr == 5 || terr == 4 {
        bonus = 1.5;
    }
    if on_enemy_city {
        bonus = bonus.max(1.5);
    }
    let df = ds.def * (du.hp / def_max.max(1.0)) * bonus;
    let total = (af + df).max(0.0001);
    let dmg = ((af / total) * us_.atk * 4.5 + 0.500001).floor();
    let our_local = ctx.my_units.iter().filter(|&&x| cheb(st.size, st.units[x].tile, a.target as usize) <= 1).count();
    let their_local = ctx.enemy_units.iter().filter(|&&x| cheb(st.size, st.units[x].tile, a.target as usize) <= 1).count();
    if dmg < du.hp {
        // no kill: only with overwhelming local superiority
        if our_local >= their_local + 2 && dmg >= du.hp * 0.6 && hp_frac > 0.6 {
            return 300.0;
        }
        return 0.0;
    }
    let mut v = 800.0;
    if on_enemy_city {
        v += 250.0;
    }
    if on_our_city {
        v += 320.0;
    }
    if ctx.my_cities.iter().any(|&c| cheb(st.size, c, a.target as usize) <= 3) {
        v += 120.0;
    }
    let retal = ((df / total) * ds.def * 4.5 + 0.500001).floor();
    if retal >= st.units[si].hp && !ds.stiff {
        if our_local > their_local + 1 {
            v -= 250.0;
        } else {
            return 0.0;
        }
    }
    if hp_frac < 0.35 && !on_our_city {
        v -= 300.0;
    }
    v
}

fn move_score(st: &St, ctx: &Ctx, a: &Act) -> f32 {
    if a.source < 0 || a.target < 0 {
        return 0.0;
    }
    let si = st.at[a.source as usize];
    if si < 0 {
        return 0.0;
    }
    let si = si as usize;
    let u = &st.units[si];
    if u.owner != st.me {
        return 0.0;
    }
    let goal = ctx.goals[si];
    if goal == usize::MAX {
        return 0.0;
    }
    let d0 = cheb(st.size, u.tile, goal);
    let d1 = cheb(st.size, a.target as usize, goal);
    let mut s = 26.0 * (d0 - d1) as f32;
    let t = &st.tiles[a.target as usize];
    if t.structure == 1 && t.owner == 0 {
        s += 240.0;
    }
    if !t.known {
        s += 18.0;
    }
    // crowd avoidance
    for &other in &ctx.my_units {
        if other != si && cheb(st.size, st.units[other].tile, a.target as usize) <= 1 {
            s -= 22.0;
            break;
        }
    }
    if d0 == 0 {
        s -= 260.0;
    }
    s
}

fn choose(obs: &Value) -> String {
    let st = parse(obs);
    let acts = parse_acts(obs);
    if acts.is_empty() {
        return String::new();
    }
    let ctx = build_ctx(&st);
    let mut best = f32::NEG_INFINITY;
    let mut bi = 0usize;
    for (i, a) in acts.iter().enumerate() {
        let s = score(&st, &ctx, a);
        if s > best {
            best = s;
            bi = i;
        }
    }
    acts[bi].id.clone()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let id = match serde_json::from_str::<Value>(&line) {
            Ok(o) => choose(&o),
            Err(_) => String::new(),
        };
        let _ = writeln!(stdout, "{}", json!({ "action_id": id }));
        let _ = stdout.flush();
    }
}
