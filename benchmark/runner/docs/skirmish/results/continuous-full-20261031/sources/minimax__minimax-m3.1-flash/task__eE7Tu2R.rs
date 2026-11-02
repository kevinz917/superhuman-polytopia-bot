// Polytopia Bardur 1v1 bot: heuristic policy over the public observation.
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

// ---------------- unit tables (Bardur-relevant kinds) ----------------
fn u_hp(k: i32) -> f32 { match k { 2=>10.0,3=>10.0,4=>10.0,5=>15.0,8=>10.0,9=>10.0,10=>10.0,11=>15.0,12=>40.0,37=>10.0,38=>5.0,39=>5.0,40=>10.0,41=>10.0,42=>10.0,43=>10.0,44=>10.0,45=>40.0, _=>10.0 } }
fn u_atk(k: i32) -> f32 { match k { 2=>2.0,3=>2.0,4=>3.5,5=>1.0,8=>4.0,9=>2.0,10=>0.0,11=>3.0,12=>5.0,37=>2.0,38=>0.0,39=>0.0,40=>2.0,41=>3.0,42=>2.0,43=>0.0,44=>3.0,45=>4.0, _=>2.0 } }
fn u_def(k: i32) -> f32 { match k { 2=>2.0,3=>1.0,4=>1.0,5=>3.0,8=>0.0,9=>1.0,10=>1.0,11=>3.0,12=>4.0,37=>2.0,38=>0.5,39=>0.5,40=>2.0,41=>2.0,42=>1.0,43=>1.0,44=>3.0,45=>4.0, _=>2.0 } }
fn u_mv(k: i32) -> i32 { match k { 2=>1,3=>2,4=>3,5=>1,8=>1,9=>1,10=>1,11=>1,12=>1,37=>1,38=>2,39=>2,40=>2,41=>2,42=>3,43=>2,44=>3,45=>2, _=>1 } }
fn u_cost(k: i32) -> i32 { match k { 2=>2,3=>3,4=>8,5=>3,8=>8,9=>3,10=>5,11=>5,12=>10,42=>5,43=>0,44=>5,41=>15, _=>2 } }
fn u_fortify(k: i32) -> bool { matches!(k, 2|3|4|5|9|17|25|37) }
fn u_stiff(k: i32) -> bool { matches!(k, 8|10|17|22|26|32|34|35|36|38|39|41|43|44|45|57|60|61|62) }
fn u_surprise(k: i32) -> bool { matches!(k, 30|37|40|62) }
fn u_value(k: i32) -> f64 { match k { 9=>28.0,8=>30.0,4=>26.0,11=>22.0,12=>35.0,5=>18.0,3=>12.0,2=>8.0,41=>30.0,44=>22.0,42=>20.0,10=>25.0,_=>10.0 } }
fn u_tech(k: i32) -> i32 { match k { 2=>0,3=>1,9=>18,11=>24,4=>3,5=>7,8=>17,42=>13,44=>44,41=>14,43=>10,10=>22,12=>-1,_=>0 } }

// research order for Bardur
const TECH_PRIORITY: &[i32] = &[1,6,18,8,9,20,23,24,2,3,4,5,7,10,44,13,16,17,21,22];

#[derive(Clone, Default)]
struct Tile { known: bool, terrain: i32, owner: i32, resource: i32, structure: i32, capital: i32, road: bool }
#[derive(Clone, Default)]
struct Unit { tile: usize, owner: i32, kind: i32, health: f32, veteran: bool, moved: bool, attacked: bool, home: i32, kills: i32 }
#[derive(Clone, Default)]
struct City { tile: usize, owner: i32, level: i32, population: i32, production: i32, progress: i32, connected: bool, riot: bool, wall: i32 }
#[derive(Clone)]
struct Act { id: String, kind: i32, source: i32, target: i32, param: i32 }

struct St {
    size: usize, n: usize, turn: i32, stars: i32, me: i32,
    tech: Vec<bool>,
    tiles: Vec<Tile>, units: Vec<Unit>, cities: Vec<City>, acts: Vec<Act>,
}

fn gi(v: &Value, k: &str) -> i32 { v.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32 }
fn gb(v: &Value, k: &str) -> bool { v.get(k).and_then(|x| x.as_bool()).unwrap_or(false) }
fn gf(v: &Value, k: &str) -> f32 { v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32 }

fn parse_state(obs: &Value) -> St {
    let vs = &obs["visible_state"];
    let size = gi(vs, "board_size").max(2) as usize;
    let n = size * size;
    let me = gi(obs, "player");
    let mut tech = vec![false; 128];
    if let Some(a) = vs["technology"].as_array() {
        for t in a { let k = t.as_i64().unwrap_or(-1); if k >= 0 && k < 128 { tech[k as usize] = true; } }
    }
    let mut tiles = vec![Tile::default(); n];
    if let Some(a) = vs["tiles"].as_array() {
        for t in a {
            let i = gi(t, "tile") as usize;
            if i >= n { continue; }
            tiles[i].known = gb(t, "known");
            if tiles[i].known {
                tiles[i].terrain = gi(t, "terrain");
                tiles[i].owner = gi(t, "owner");
                tiles[i].resource = gi(t, "resource");
                tiles[i].structure = gi(t, "structure");
                tiles[i].capital = gi(t, "capital");
                tiles[i].road = gi(t, "road") != 0;
            }
        }
    }
    let mut units: Vec<Unit> = Vec::new();
    if let Some(a) = vs["units"].as_array() {
        for u in a {
            units.push(Unit {
                tile: gi(u, "tile") as usize, owner: gi(u, "owner"), kind: gi(u, "kind"),
                health: gf(u, "health"), veteran: gb(u, "veteran"), moved: gb(u, "moved"),
                attacked: gb(u, "attacked"), home: gi(u, "home"), kills: gi(u, "kills") });
        }
    }
    let mut cities: Vec<City> = Vec::new();
    if let Some(a) = vs["cities"].as_array() {
        for c in a {
            let mut city = City { tile: gi(c, "tile") as usize, owner: gi(c, "owner"), level: gi(c, "level"),
                population: gi(c, "population"), production: gi(c, "production"), progress: gi(c, "progress"),
                connected: gb(c, "connected"), riot: gb(c, "riot"), wall: 0 };
            if let Some(r) = c["rewards"].as_array() { if r.len() > 1 { city.wall = gi(&r[1], ""); } }
            if city.wall == 0 { if let Some(r) = c["rewards"].as_array() { if r.len() > 1 { city.wall = r[1].as_i64().unwrap_or(0) as i32; } } }
            cities.push(city);
        }
    }
    let mut acts: Vec<Act> = Vec::new();
    if let Some(a) = obs["legal_actions"].as_array() {
        for e in a {
            let c = &e["command"];
            acts.push(Act { id: e["action_id"].as_str().unwrap_or("").to_string(),
                kind: gi(c, "kind"), source: gi(c, "source"), target: gi(c, "target"), param: gi(c, "parameter") });
        }
    }
    St { size, n, turn: gi(vs, "turn"), stars: gi(vs, "stars"), me, tech, tiles, units, cities, acts }
}

fn dist(a: usize, b: usize, size: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

// combat model (engine parity): returns (damage to defender, retaliation to attacker)
fn combat(atk: f32, ah: f32, am: f32, df: f32, dh: f32, dm: f32, bonus: f32) -> (f32, f32) {
    let af = atk * (ah / am.max(1.0));
    let dff = df * (dh / dm.max(1.0)) * bonus;
    let total = af + dff;
    if total <= 0.0 { return (0.0, 0.0); }
    let a = ((af / total) * atk * 4.5 + 0.000001 + 0.5).floor();
    let d = if dh > a { ((dff / total) * df * 4.5 + 0.000001 + 0.5).floor() } else { 0.0 };
    (a, d)
}

struct Mem { last_turn: i32, acts: u32 }
impl Mem { fn new() -> Mem { Mem { last_turn: -1, acts: 0 } } }

// ---------------- context ----------------
struct Ctx {
    my_cities: Vec<usize>, enemy_cities: Vec<usize>, villages: Vec<usize>,
    my_units: Vec<usize>, enemy_units: Vec<usize>,
    goals: Vec<Vec<(usize, f64)>>,
    army: usize, ncity: usize, capacity: usize,
    unit_at: Vec<usize>, city_at: Vec<i32>,
}

fn tech_tier(t: i32) -> i32 { match t { 1|6|7|10|15|20|21|44|48 => 1, 2|4|8|13|16|18|23|38 => 2, 3|5|9|12|14|17|19|22|24 => 3, _ => 1 } }

fn build_ctx(st: &St) -> Ctx {
    let mut ctx = Ctx { my_cities: vec![], enemy_cities: vec![], villages: vec![], my_units: vec![], enemy_units: vec![],
        goals: vec![Vec::new(); st.units.len()], army: 0, ncity: 0, capacity: 0,
        unit_at: vec![usize::MAX; st.n], city_at: vec![-1; st.n] };
    for (i, c) in st.cities.iter().enumerate() {
        if c.tile < st.n { ctx.city_at[c.tile] = i as i32; }
        if c.owner == st.me { ctx.my_cities.push(i); ctx.ncity += 1; ctx.capacity += c.level.max(1) as usize; }
        else if c.owner > 0 { ctx.enemy_cities.push(i); }
        else { ctx.villages.push(i); }
    }
    for (i, u) in st.units.iter().enumerate() {
        if u.tile < st.n { ctx.unit_at[u.tile] = i; }
        if u.owner == st.me { ctx.my_units.push(i); ctx.army += 1; }
        else if u.owner > 0 { ctx.enemy_units.push(i); }
    }
    // frontier candidates: known tiles bordering unknown area, ranked by how much unknown land is near
    let size = st.size as i32;
    let mut frontier: Vec<(usize, i32)> = Vec::new();
    for t in 0..st.n {
        if !st.tiles[t].known { continue; }
        let (cx, cy) = ((t % st.size) as i32, (t / st.size) as i32);
        let mut isf = false; let mut unk = 0;
        for y in (cy - 2).max(0)..(cy + 3).min(size) {
            for x in (cx - 2).max(0)..(cx + 3).min(size) {
                let i = (y * size + x) as usize;
                if !st.tiles[i].known { unk += 1; if (x - cx).abs() <= 1 && (y - cy).abs() <= 1 { isf = true; } }
            }
        }
        if isf { frontier.push((t, unk)); }
    }
    frontier.sort_by(|a, b| b.1.cmp(&a.1));
    frontier.truncate(16);
    // goals
    let mut load = vec![0i32; st.n];
    for (rank, &ui) in ctx.my_units.iter().enumerate() {
        let u = &st.units[ui];
        let mut goals: Vec<(usize, f64)> = Vec::new();
        // 1. enemy units that menace one of our cities (defence)
        let mut threat: Option<(usize, i32)> = None;
        for &ei in ctx.enemy_units.iter() {
            let e = &st.units[ei];
            let d = dist(u.tile, e.tile, st.size);
            let near = ctx.my_cities.iter().any(|&ci| dist(st.cities[ci].tile, e.tile, st.size) <= 2 && st.cities[ci].level >= 2);
            if d <= 3 && near && threat.map(|(_, bd)| d < bd).unwrap_or(true) {
                threat = Some((ei, d));
            }
        }
        if let Some((ei, _)) = threat { goals.push((st.units[ei].tile, 140.0)); }
        // 2. an enemy standing next to this unit
        let mut adj: Option<usize> = None;
        for &ei in ctx.enemy_units.iter() {
            let d = dist(u.tile, st.units[ei].tile, st.size);
            if d <= 1 { adj = Some(ei); break; }
        }
        if let Some(ei) = adj { goals.push((st.units[ei].tile, 55.0)); }
        // retreat to the nearest own city when badly hurt
        if u.health < u_hp(u.kind) * 0.45 {
            let mut best = usize::MAX; let mut bd = i32::MAX;
            for &ci in ctx.my_cities.iter() { let d = dist(u.tile, st.cities[ci].tile, st.size); if d < bd { bd = d; best = ci; } }
            if best != usize::MAX { goals.push((st.cities[best].tile, 130.0)); }
        }
        // 3. expand to the nearest neutral village (spread over units)
        if ctx.ncity < 8 && !ctx.villages.is_empty() {
            let mut best = usize::MAX; let mut bestc = 1e9f64;
            for &ci in ctx.villages.iter() {
                let vt = st.cities[ci].tile;
                let d = dist(u.tile, vt, st.size);
                if d > 12 { continue; }
                let c = d as f64 + 4.0 * load[vt] as f64;
                if c < bestc { bestc = c; best = vt; }
            }
            if best != usize::MAX { load[best] += 1; goals.push((best, 170.0)); }
        }
        // 4. attack a known enemy city with a real army
        if !ctx.enemy_cities.is_empty() && ctx.army >= 3 {
            let mut best = usize::MAX; let mut bd = i32::MAX;
            for &ci in ctx.enemy_cities.iter() {
                let d = dist(u.tile, st.cities[ci].tile, st.size);
                if d < bd { bd = d; best = ci; }
            }
            if best != usize::MAX && bd <= 14 { goals.push((st.cities[best].tile, 70.0)); }
        }
        // 5. explore
        if !frontier.is_empty() {
            goals.push((frontier[rank % frontier.len()].0, 10.0));
        }
        ctx.goals[ui] = goals;
    }
    ctx
}

fn near_city(st: &St, ctx: &Ctx, tile: usize, rad: i32) -> bool {
    ctx.my_cities.iter().any(|&ci| dist(st.cities[ci].tile, tile, st.size) <= rad)
}

fn def_bonus(st: &St, ctx: &Ctx, them: &Unit) -> f32 {
    let t = &st.tiles[them.tile];
    let ci = ctx.city_at[them.tile];
    if ci >= 0 {
        let c = &st.cities[ci as usize];
        if c.owner == them.owner && u_fortify(them.kind) { return if c.wall > 0 { 4.0 } else { 1.5 }; }
        if c.owner == them.owner { return 1.0; }
    }
    match t.terrain { 4 => 1.5, 5 => 1.5, _ => 1.0 }
}

fn score(st: &St, ctx: &Ctx, a: &Act) -> f64 {
    match a.kind {
        9 => { // city reward
            let mut base = match a.param {
                6 => 85.0, // giant
                8 => 95.0, // +3 population
                3 => 70.0, // workshop: +1 production
                2 => 68.0, // park: +1 production
                7 => 80.0, // +5 stars
                5 => 45.0, // border growth
                1 => 55.0, // wall
                4 => 30.0, // explorer
                _ => 10.0 };
            if a.param == 7 && st.stars < 10 { base += 25.0; }
            if a.param == 1 {
                let threatened = ctx.enemy_units.iter().any(|&ei| near_city(st, &ctx, st.units[ei].tile, 3));
                if threatened { base += 60.0; }
            }
            base
        }
        8 => { // capture
            let t = a.source;
            if t < 0 || t as usize >= st.n { return -1e9; }
            let tl = &st.tiles[t as usize];
            if tl.structure == 1 {
                if tl.owner == 0 { 320.0 } else if tl.owner != st.me { 6000.0 } else { -1e9 }
            } else if tl.structure == 2 { 260.0 } else { 90.0 }
        }
        2 => { // attack
            let si = a.source; let ti = a.target;
            if si < 0 || ti < 0 { return -1e9; }
            let ui = ctx.unit_at[si as usize]; let xi = ctx.unit_at[ti as usize];
            if ui == usize::MAX || xi == usize::MAX { return -1e9; }
            let me = &st.units[ui]; let them = &st.units[xi];
            if them.owner == st.me { return -1e9; }
            let bonus = def_bonus(st, ctx, them);
            let (dmg, ret) = combat(u_atk(me.kind), me.health, u_hp(me.kind), u_def(them.kind),
                them.health, u_hp(them.kind), bonus);
            let ret = if u_stiff(them.kind) || u_surprise(me.kind) { 0.0 } else { ret };
            let kill = dmg >= them.health;
            if kill {
                if ret >= me.health { 130.0 + u_value(them.kind) * 0.5 } else { 340.0 + u_value(them.kind) }
            } else {
                if ret >= me.health { return -5000.0; }
                let mut s = (dmg - ret) as f64 * 16.0;
                if near_city(st, ctx, them.tile, 1) { s += 120.0; } else if near_city(st, ctx, them.tile, 3) { s += 50.0; }
                if me.health < u_hp(me.kind) * 0.6 { s -= 120.0; }
                if s < 0.0 { -700.0 } else { s }
            }
        }
        3 => match a.param {
            16 => 45.0,          // promote
            9 => 25.0,           // heal others
            7 => 6.0,            // recover
            2 => 12.0,           // clear forest (+1 star)
            1 => -50.0, 3 => -50.0,
            8 => -80.0,          // disband
            4 => -50.0,          // destroy
            18..=22 => -1e9,     // diplomacy
            _ => -100.0 },
        4 => { // summon
            let k = a.param;
            if u_tech(k) >= 0 && !st.tech[u_tech(k) as usize] && k != 2 { return -1e9; }
            let desired = (2 + 2 * ctx.ncity).min(20);
            let mut s = 160.0;
            s += match k { 9 => 18.0, 11 => 24.0, 4 => 22.0, 3 => 16.0, 5 => 16.0, 2 => 8.0, 12 => 30.0, _ => 0.0 };
            if ctx.army >= desired { s -= 90.0; }
            if st.stars < u_cost(k) * 2 { s -= 30.0; }
            s
        }
        5 => { // harvest
            let t = a.target;
            if t < 0 || t as usize >= st.n { return -1e9; }
            let mut s = 100.0;
            if st.stars < 5 { s -= 40.0; }
            s
        }
        6 => { // build
            match a.param {
                5 => 85.0,   // farm
                6 => 80.0,   // windmill
                50 => 85.0,  // market
                8 => 70.0,   // port
                13 => 60.0,  // sawmill
                3 => 48.0,   // road
                12 => 35.0,  // lumber hut
                48 => 40.0,  // bridge
                17|18|19|20 => 25.0,
                _ => -1e9 }
        }
        7 => { // research
            let idx = TECH_PRIORITY.iter().position(|&t| t == a.param);
            let mut s = match idx { Some(i) => 34.0 - i as f64 * 1.2, None => 6.0 };
            let cities = ctx.ncity as i32;
            let cost = 4 + cities * tech_tier(a.param);
            if st.stars - cost < 12 { s -= 60.0; }
            if ctx.army >= ctx.capacity { s -= 60.0; }
            if ctx.ncity <= 1 { s -= 40.0; }
            s
        }
        1 => { // move
            let si = a.source; let ti = a.target;
            if si < 0 || ti < 0 { return -1e9; }
            let ui = ctx.unit_at[si as usize];
            if ui == usize::MAX { return -1e9; }
            let cur = st.units[ui].tile;
            let mut best = 0.0f64;
            for &(g, w) in ctx.goals[ui].iter() {
                let imp = (dist(cur, g, st.size) - dist(ti as usize, g, st.size)) as f64;
                if imp > 0.0 { let v = w * imp; if v > best { best = v; } }
            }
            best
        }
        10 => -1e6,
        11 => -1e9,
        _ => -1e9,
    }
}

fn decide(st: &St, mem: &mut Mem) -> String {
    if st.acts.is_empty() { return String::new(); }
    let ctx = build_ctx(st);
    let mut best = -1e9f64; let mut bidx = 0usize;
    for (i, a) in st.acts.iter().enumerate() {
        let s = score(st, &ctx, a);
        if s > best { best = s; bidx = i; }
    }
    if mem.acts >= 26 {
        if let Some(i) = st.acts.iter().position(|a| a.kind == 10) { return st.acts[i].id.clone(); }
    }
    if best < -100.0 {
        if let Some(i) = st.acts.iter().position(|a| a.kind == 10) { return st.acts[i].id.clone(); }
    }
    st.acts[bidx].id.clone()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = Mem::new();
    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let obs: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };
        let st = parse_state(&obs);
        if st.turn != mem.last_turn { mem.last_turn = st.turn; mem.acts = 0; }
        let id = decide(&st, &mut mem);
        mem.acts += 1;
        writeln!(stdout, "{}", json!({ "action_id": id })).unwrap();
        stdout.flush().unwrap();
    }
}
