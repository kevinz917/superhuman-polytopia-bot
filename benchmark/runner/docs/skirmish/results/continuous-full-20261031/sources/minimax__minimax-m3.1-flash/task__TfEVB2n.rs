// Polytopia Skirmish bot: Bardur 1v1 heuristic policy.
// One JSON observation per line; reply {"action_id":"..."} and flush.
use serde_json::{json, Value};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{self, BufRead, Write};

const INF: i32 = 1_000_000;

#[derive(Clone, Copy, Default)]
struct Tile {
    known: bool,
    terrain: i32,
    owner: i32,
    resource: i32,
    structure: i32,
    road: bool,
    capital: i32,
}

#[derive(Clone, Copy)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
    moved: bool,
    attacked: bool,
    kills: i32,
    home: i32,
    veteran: bool,
}

#[derive(Clone, Copy)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    population: i32,
    production: i32,
    progress: i32,
    wall: bool,
    rewards: i32,
}

struct Game {
    size: usize,
    n: usize,
    player: i32,
    other: i32,
    turn: i32,
    stars: i32,
    tech: Vec<bool>,
    tiles: Vec<Tile>,
    units: Vec<Unit>,
    cities: Vec<City>,
    blocker: Vec<i32>,
}

// (health, attack, defense, movement, range, star cost)
fn unit_stats(kind: i32) -> (f32, f32, f32, i32, i32, i32) {
    match kind {
        2 => (10.0, 2.0, 2.0, 1, 1, 2),
        3 => (10.0, 2.0, 1.0, 2, 1, 3),
        4 => (10.0, 3.5, 1.0, 3, 1, 8),
        5 => (15.0, 1.0, 3.0, 1, 1, 3),
        8 => (10.0, 4.0, 0.0, 1, 3, 8),
        9 => (10.0, 2.0, 1.0, 1, 2, 3),
        10 => (10.0, 0.0, 1.0, 1, 1, 5),
        11 => (15.0, 3.0, 3.0, 1, 1, 5),
        12 => (40.0, 5.0, 4.0, 1, 1, 10),
        37 => (10.0, 2.0, 2.0, 1, 1, 2),
        38 => (5.0, 0.0, 0.5, 2, 1, 8),
        39 => (5.0, 0.0, 0.5, 2, 1, 2),
        40 => (10.0, 2.0, 2.0, 2, 1, 2),
        41 => (10.0, 3.0, 2.0, 2, 3, 15),
        42 => (10.0, 2.0, 1.0, 3, 2, 5),
        43 => (10.0, 0.0, 1.0, 2, 2, 0),
        44 => (10.0, 3.0, 3.0, 3, 1, 5),
        45 => (40.0, 4.0, 4.0, 2, 1, 10),
        _ => (10.0, 1.0, 1.0, 1, 1, 5),
    }
}

fn is_float(kind: i32) -> bool {
    matches!(kind, 39 | 40 | 41 | 42 | 43 | 44 | 45)
}
fn is_dash(kind: i32) -> bool {
    matches!(kind, 2 | 3 | 4 | 9 | 11 | 37 | 38 | 40)
}
fn is_fortify(kind: i32) -> bool {
    matches!(kind, 2 | 3 | 4 | 5 | 9)
}

fn parse(obs: &Value) -> Game {
    let v = &obs["visible_state"];
    let size = v["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let player = obs["player"].as_i64().unwrap_or(1) as i32;
    let mut tech = vec![false; 128];
    if let Some(arr) = v["technology"].as_array() {
        for t in arr {
            if let Some(k) = t.as_u64() {
                if (k as usize) < 128 {
                    tech[k as usize] = true;
                }
            }
        }
    }
    let mut tiles = vec![Tile::default(); n];
    if let Some(arr) = v["tiles"].as_array() {
        for t in arr {
            let i = t["tile"].as_u64().unwrap_or(0) as usize;
            if i >= n {
                continue;
            }
            if t["known"].as_bool().unwrap_or(false) {
                tiles[i].known = true;
                tiles[i].terrain = t["terrain"].as_i64().unwrap_or(0) as i32;
                tiles[i].owner = t["owner"].as_i64().unwrap_or(0) as i32;
                tiles[i].resource = t["resource"].as_i64().unwrap_or(0) as i32;
                tiles[i].structure = t["structure"].as_i64().unwrap_or(0) as i32;
                tiles[i].road = t["road"].as_i64().unwrap_or(0) != 0;
                tiles[i].capital = t["capital"].as_i64().unwrap_or(0) as i32;
            }
        }
    }
    let mut units: Vec<Unit> = Vec::new();
    let mut blocker = vec![0i32; n];
    if let Some(arr) = v["units"].as_array() {
        for u in arr {
            let tile = u["tile"].as_u64().unwrap_or(0) as usize;
            if tile >= n {
                continue;
            }
            let owner = u["owner"].as_i64().unwrap_or(0) as i32;
            let un = Unit {
                tile,
                owner,
                kind: u["kind"].as_i64().unwrap_or(0) as i32,
                health: u["health"].as_f64().unwrap_or(0.0) as f32,
                moved: u["moved"].as_bool().unwrap_or(false),
                attacked: u["attacked"].as_bool().unwrap_or(false),
                kills: u["kills"].as_i64().unwrap_or(0) as i32,
                home: u["home"].as_i64().unwrap_or(0) as i32,
                veteran: u["veteran"].as_bool().unwrap_or(false),
            };
            blocker[tile] = owner;
            units.push(un);
        }
    }
    let mut cities: Vec<City> = Vec::new();
    if let Some(arr) = v["cities"].as_array() {
        for c in arr {
            let tile = c["tile"].as_u64().unwrap_or(0) as usize;
            if tile >= n {
                continue;
            }
            let owner = c["owner"].as_i64().unwrap_or(0) as i32;
            let mut wall = false;
            let mut rewards = 0i32;
            if let Some(r) = c["rewards"].as_array() {
                if r.len() > 1 {
                    wall = r[1].as_i64().unwrap_or(0) > 0;
                }
                for x in r {
                    rewards += x.as_i64().unwrap_or(0) as i32;
                }
            }
            cities.push(City {
                tile,
                owner,
                level: c["level"].as_i64().unwrap_or(1) as i32,
                population: c["population"].as_i64().unwrap_or(0) as i32,
                production: c["production"].as_i64().unwrap_or(1) as i32,
                progress: c["progress"].as_i64().unwrap_or(0) as i32,
                wall,
                rewards,
            });
        }
    }
    Game {
        size,
        n,
        player,
        other: 3 - player,
        turn: v["turn"].as_i64().unwrap_or(0) as i32,
        stars: v["stars"].as_i64().unwrap_or(0) as i32,
        tech,
        tiles,
        units,
        cities,
        blocker,
    }
}

fn cheb(g: &Game, a: usize, b: usize) -> i32 {
    let ax = (a % g.size) as i32;
    let ay = (a / g.size) as i32;
    let bx = (b % g.size) as i32;
    let by = (b / g.size) as i32;
    (ax - bx).abs().max((ay - by).abs())
}

// Cost-to-goal field over known tiles (Dijkstra backwards from the goal).
fn path_field(g: &Game, goal: usize, float: bool) -> Vec<i32> {
    let n = g.n;
    let mut dist = vec![INF; n];
    let mut heap: BinaryHeap<Reverse<(i32, usize)>> = BinaryHeap::new();
    dist[goal] = 0;
    heap.push(Reverse((0, goal)));
    while let Some(Reverse((d, u))) = heap.pop() {
        if d > dist[u] {
            continue;
        }
        let ux = (u % g.size) as i32;
        let uy = (u / g.size) as i32;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let x = ux + dx;
                let y = uy + dy;
                if x < 0 || y < 0 || x >= g.size as i32 || y >= g.size as i32 {
                    continue;
                }
                let v = (y * g.size as i32 + x) as usize;
                if !g.tiles[v].known {
                    continue;
                }
                if g.blocker[v] == g.other {
                    continue;
                }
                let t = g.tiles[v];
                let water = t.terrain == 1 || t.terrain == 2;
                let pass = if water {
                    float || t.structure == 48 || (t.structure == 8 && t.owner == g.player)
                } else {
                    t.terrain != 4 || g.tech[20]
                };
                if !pass {
                    continue;
                }
                let mut c = 1;
                if t.terrain == 5 && !t.road {
                    c = 3;
                }
                if t.terrain == 4 {
                    c = 5;
                }
                if water {
                    c = 2;
                }
                let nd = d + c;
                if nd < dist[v] {
                    dist[v] = nd;
                    heap.push(Reverse((nd, v)));
                }
            }
        }
    }
    dist
}

struct Ctx {
    spare: i32,
    threatened: bool,
    animals: i32,
    my_units: Vec<usize>,
    enemy_units: Vec<usize>,
    my_cities: Vec<usize>,
    enemy_cities: Vec<usize>,
    neutral_villages: Vec<usize>,
    my_capital: Option<usize>,
    enemy_capital: Option<usize>,
}

fn build_ctx(g: &Game) -> Ctx {
    let mut my_units = Vec::new();
    let mut enemy_units = Vec::new();
    for (i, u) in g.units.iter().enumerate() {
        if u.owner == g.player {
            my_units.push(i);
        } else if u.owner == g.other {
            enemy_units.push(i);
        }
    }
    let mut my_cities = Vec::new();
    let mut enemy_cities = Vec::new();
    let mut neutral_villages = Vec::new();
    for (i, c) in g.cities.iter().enumerate() {
        if c.owner == g.player {
            my_cities.push(i);
        } else if c.owner == g.other {
            enemy_cities.push(i);
        } else {
            neutral_villages.push(i);
        }
    }
    let mut my_capital = None;
    let mut enemy_capital = None;
    for t in 0..g.n {
        if !g.tiles[t].known {
            continue;
        }
        if g.tiles[t].capital == g.player {
            my_capital = Some(t);
        } else if g.tiles[t].capital == g.other {
            enemy_capital = Some(t);
        }
    }
    let mut cap = 0i32;
    for &ci in &my_cities {
        cap += g.cities[ci].level;
    }
    let mut threatened = false;
    for &ei in &enemy_units {
        let et = g.units[ei].tile;
        for &ci in &my_cities {
            if cheb(g, g.cities[ci].tile, et) <= 6 {
                threatened = true;
            }
        }
    }
    let mut animals = 0;
    for t in 0..g.n {
        if g.tiles[t].known && g.tiles[t].owner == g.player && g.tiles[t].resource == 1 {
            animals += 1;
        }
    }
    Ctx {
        spare: cap - my_units.len() as i32,
        threatened,
        animals,
        my_units,
        enemy_units,
        my_cities,
        enemy_cities,
        neutral_villages,
        my_capital,
        enemy_capital,
    }
}

// Assign a destination goal and a role to each of our units.
// Roles: 0 = expand (capture a village), 1 = attack a known enemy city,
//        2 = explore, 3 = defend (hold the capital).
fn assign_goals(g: &Game, ctx: &Ctx) -> (Vec<Option<usize>>, Vec<u8>) {
    let mut goals: Vec<Option<usize>> = vec![None; g.units.len()];
    let mut roles: Vec<u8> = vec![3; g.units.len()];
    let mut assigned = vec![false; g.units.len()];
    let my_units = &ctx.my_units;
    let cap = ctx.my_capital;

    let mut workers = if my_units.len() > 2 { my_units.len() - 2 } else { 1 };
    if workers == 0 && !my_units.is_empty() {
        workers = 1;
    }
    if workers > 4 {
        workers = 4;
    }
    if ctx.my_cities.len() >= 9 {
        workers = workers.min(2);
    }
    let attacking = ctx.enemy_capital.is_some() && my_units.len() >= 10;

    // 1. Expansion workers take the nearest neutral villages.
    if workers > 0 && !ctx.neutral_villages.is_empty() {
        let mut villages: Vec<(usize, i32, i32)> = Vec::new();
        for &vi in &ctx.neutral_villages {
            let vt = g.cities[vi].tile;
            let mut d = i32::MAX;
            for &ci in &ctx.my_cities {
                d = d.min(cheb(g, g.cities[ci].tile, vt));
            }
            if d == i32::MAX {
                d = cheb(g, cap.unwrap_or(vt), vt);
            }
            let mut risk = 0;
            for &ei in &ctx.enemy_units {
                if cheb(g, g.units[ei].tile, vt) <= 2 {
                    risk += 1;
                }
            }
            villages.push((vt, d, risk));
        }
        villages.sort_by_key(|&(_, d, r)| (r, d));
        let mut taken = 0;
        for (vt, _, _) in villages {
            if taken >= workers {
                break;
            }
            let mut best: Option<(usize, i32)> = None;
            for &ui in my_units {
                if assigned[ui] {
                    continue;
                }
                let d = cheb(g, g.units[ui].tile, vt);
                if best.is_none() || d < best.unwrap().1 {
                    best = Some((ui, d));
                }
            }
            if let Some((ui, _)) = best {
                assigned[ui] = true;
                roles[ui] = 0;
                goals[ui] = Some(vt);
                taken += 1;
            }
        }
    }

    // 2. Attack force: with a real army and a known capital, everybody else marches.
    if attacking {
        let ec = ctx.enemy_capital.unwrap();
        for &ui in my_units {
            if !assigned[ui] {
                assigned[ui] = true;
                roles[ui] = 1;
                goals[ui] = Some(ec);
            }
        }
        return (goals, roles);
    }

    // 3. One explorer sweeps the unknown.
    let frontier = exploration_frontier(g);
    if !frontier.is_empty() && my_units.len() >= 3 {
        let mut used: Vec<usize> = Vec::new();
        for &ui in my_units {
            if assigned[ui] {
                continue;
            }
            let mut best: Option<(usize, f32)> = None;
            for &f in &frontier {
                if used.contains(&f) {
                    continue;
                }
                let s = cheb(g, g.units[ui].tile, f) as f32 - 0.5 * cheb(g, cap.unwrap_or(f), f) as f32;
                if best.is_none() || s < best.unwrap().1 {
                    best = Some((f, s));
                }
            }
            let pick = match best {
                Some((f, _)) => f,
                None => frontier[0],
            };
            used.push(pick);
            assigned[ui] = true;
            roles[ui] = 2;
            goals[ui] = Some(pick);
            break;
        }
    }

    // 4. Everyone else holds the capital.
    for &ui in my_units {
        if !assigned[ui] {
            assigned[ui] = true;
            roles[ui] = 3;
            goals[ui] = cap;
        }
    }
    (goals, roles)
}

// Known tiles next to unknown tiles: the places worth walking to reveal more.
fn exploration_frontier(g: &Game) -> Vec<usize> {
    let mut out = Vec::new();
    for t in 0..g.n {
        if !g.tiles[t].known {
            continue;
        }
        let (x, y) = ((t % g.size) as i32, (t / g.size) as i32);
        let mut near_unknown = false;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= g.size as i32 || ny >= g.size as i32 {
                    continue;
                }
                let v = (ny * g.size as i32 + nx) as usize;
                if !g.tiles[v].known {
                    near_unknown = true;
                }
            }
        }
        if near_unknown {
            out.push(t);
        }
    }
    out
}

struct Fields {
    land: Vec<(usize, Vec<i32>)>,
    float: Vec<(usize, Vec<i32>)>,
}

impl Fields {
    fn get(&self, goal: usize, float: bool) -> &Vec<i32> {
        let v = if float { &self.float } else { &self.land };
        for (g, f) in v.iter() {
            if *g == goal {
                return f;
            }
        }
        unreachable!()
    }
}

fn build_fields(g: &Game, goals: &[Option<usize>]) -> Fields {
    let mut land = Vec::new();
    let mut flt = Vec::new();
    let mut seen: Vec<usize> = Vec::new();
    let mut need_float = false;
    for (i, o) in goals.iter().enumerate() {
        if let Some(gl) = *o {
            if !seen.contains(&gl) {
                seen.push(gl);
                land.push((gl, path_field(g, gl, false)));
                if is_float(g.units[i].kind) {
                    need_float = true;
                }
            }
        }
    }
    if need_float {
        for &gl in &seen {
            flt.push((gl, path_field(g, gl, true)));
        }
    }
    Fields { land, float: flt }
}

fn unit_at(g: &Game, tile: usize) -> Option<usize> {
    g.units.iter().position(|u| u.tile == tile)
}

fn tech_tier(t: i32) -> i32 {
    match t {
        1 | 6 | 10 | 15 | 20 | 25 | 26 | 28 | 30 | 31 | 35 | 36 | 37 | 41 | 42 | 48 | 49 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 1,
    }
}

fn tech_pref(t: i32, sea: bool, _capital_unknown: bool) -> f64 {
    match t {
        6 => 100.0,
        8 => 94.0,
        7 => 84.0,
        18 => 82.0,
        1 => 70.0,
        10 => {
            if sea {
                95.0
            } else {
                30.0
            }
        }
        7 => 80.0,
        8 => 76.0,
        2 => 72.0,
        3 => 68.0,
        4 => 58.0,
        20 => 20.0,
        23 => 42.0,
        24 => 40.0,
        44 => 34.0,
        13 => 30.0,
        14 => 24.0,
        9 => 18.0,
        15 => 5.0,
        _ => 10.0,
    }
}

fn def_bonus(g: &Game, tile: usize, def: &Unit) -> f32 {
    let t = &g.tiles[tile];
    let city_bonus = if is_fortify(def.kind) {
        let mut c = 1.0f32;
        for ci in &g.cities {
            if ci.tile == tile && ci.owner == def.owner {
                c = if ci.wall { 4.0 } else { 1.5 };
            }
        }
        c
    } else {
        1.0
    };
    match t.terrain {
        1 | 2 => {
            if g.tech[12] {
                1.5
            } else {
                1.0
            }
        }
        5 => {
            if g.tech[18] {
                1.5
            } else {
                1.0
            }
        }
        4 => {
            if g.tech[20] {
                1.5
            } else {
                1.0
            }
        }
        _ => city_bonus,
    }
}

fn combat(g: &Game, atk: &Unit, def: &Unit) -> (f32, f32) {
    let (ahp, aatk, _, _, _, _) = unit_stats(atk.kind);
    let (dmax, _, ddef, _, _, _) = unit_stats(def.kind);
    let amax = ahp + if atk.veteran { 5.0 } else { 0.0 };
    let dmx = dmax + if def.veteran { 5.0 } else { 0.0 };
    let db = def_bonus(g, def.tile, def);
    let af = aatk * (atk.health / amax.max(1.0));
    let df = ddef * (def.health / dmx.max(1.0)) * db;
    let total = af + df;
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    let damage = ((af / total) * aatk * 4.5 + 0.000001 + 0.5).floor();
    let retal = if def.health > damage {
        ((df / total) * ddef * 4.5 + 0.000001 + 0.5).floor()
    } else {
        0.0
    };
    (damage, retal)
}

fn struct_cost(k: i32) -> i32 {
    match k {
        3 => 3,
        5 => 5,
        6 => 5,
        8 => 7,
        12 => 3,
        13 => 5,
        17 => 20,
        21 => 5,
        22 => 5,
        48 => 5,
        _ => 5,
    }
}

fn score_action(g: &Game, ctx: &Ctx, goals: &[Option<usize>], roles: &[u8], fields: &Fields, cmd: &Value, sea: bool) -> f64 {
    let kind = cmd["kind"].as_i64().unwrap_or(0) as i32;
    let src = cmd["source"].as_i64().unwrap_or(-1);
    let dst = cmd["target"].as_i64().unwrap_or(-1);
    let par = cmd["parameter"].as_i64().unwrap_or(0) as i32;
    let capital_unknown = ctx.enemy_capital.is_none();
    match kind {
        9 => {
            // city reward: the slot decides which pair is offered
            let slot = ctx
                .my_cities
                .iter()
                .find(|&&ci| g.cities[ci].tile == dst as usize)
                .map(|&ci| g.cities[ci].rewards + 1)
                .unwrap_or(1);
            let p = match (slot, par) {
                (1, 3) => 76.0,
                (1, 4) => {
                    if capital_unknown && g.turn < 100 {
                        95.0
                    } else {
                        60.0
                    }
                }
                (2, 1) => {
                    if ctx.threatened {
                        97.0
                    } else {
                        45.0
                    }
                }
                (2, 7) => {
                    if ctx.threatened {
                        55.0
                    } else {
                        96.0
                    }
                }
                (3, 5) => {
                    if ctx.animals <= 1 {
                        93.0
                    } else {
                        72.0
                    }
                }
                (3, 8) => {
                    if ctx.animals <= 1 {
                        74.0
                    } else {
                        96.0
                    }
                }
                (_, 2) => 60.0,
                (_, 6) => 92.0,
                _ => 60.0,
            };
            10000.0 + p
        }
        8 => {
            // capture: the unit stands on the tile
            let tile = src as usize;
            if tile >= g.n {
                return -1e9;
            }
            let owner = g.tiles[tile].owner;
            let mut v = 0.0;
            if owner == g.other {
                v += 500.0; // enemy city: must take it
            } else {
                v += 120.0; // neutral village
            }
            // risk: enemies adjacent
            for &ei in &ctx.enemy_units {
                if cheb(g, g.units[ei].tile, tile) <= 1 {
                    v -= 120.0;
                }
            }
            5000.0 + v
        }
        2 => {
            // attack
            let ai = match unit_at(g, src as usize) {
                Some(i) => i,
                None => return -1e9,
            };
            let di = match unit_at(g, dst as usize) {
                Some(i) => i,
                None => return -1e9,
            };
            let atk = &g.units[ai];
            let def = &g.units[di];
            if def.owner != g.other {
                return -1e9;
            }
            let (dmg, ret) = combat(g, atk, def);
            let kills = def.health - dmg <= 0.0;
            let we_die = atk.health - ret <= 0.0;
            let dcost = unit_stats(def.kind).5 as f64;
            let acost = unit_stats(atk.kind).5 as f64;
            let mut s = if kills && !we_die {
                420.0 + 25.0 * dcost
            } else if kills && we_die {
                -200.0 + 25.0 * (dcost - acost)
            } else if !kills && we_die {
                -2000.0
            } else {
                60.0 + 12.0 * dmg as f64 - 10.0 * ret as f64
            };
            if g.tiles[def.tile].owner == g.player {
                s += 120.0; // defending our land
            }
            let mut urgent = false;
            for &ci in &ctx.my_cities {
                if cheb(g, g.cities[ci].tile, def.tile) <= 1 {
                    s += 200.0; // it is next to one of our cities
                    urgent = true;
                }
            }
            let role = roles.get(ai).copied().unwrap_or(3);
            if role == 0 || role == 2 {
                // expansion workers and the scout do not pick fights
                if !urgent && !(kills && !we_die) {
                    return -500.0;
                }
                if urgent {
                    s += 300.0;
                } else {
                    s += 100.0;
                }
            } else if role == 3 {
                s += 250.0; // defenders hit back
            }
            for &ci in &ctx.enemy_cities {
                if g.cities[ci].tile == def.tile {
                    s += 60.0;
                }
            }
            2000.0 + s
        }
        7 => {
            // research
            let cities = ctx.my_cities.len() as i32;
            let mut cost = 4 + cities * tech_tier(par);
            if g.tech[22] {
                cost = ((cost as f32) * 0.66666).ceil() as i32;
            }
            let reserve = 4 + cities / 2;
            if g.stars - cost < reserve {
                return -2000.0;
            }
            1000.0 + tech_pref(par, sea, capital_unknown) - 3.0 * cost as f64
        }
        4 => {
            // summon
            let cost = unit_stats(par).5;
            if g.stars - cost < 2 {
                return -2000.0;
            }
            if ctx.my_units.len() >= 18 {
                return -2000.0;
            }
            if ctx.spare <= 0 {
                return -2000.0;
            }
            let p = match par {
                43 => {
                    if sea {
                        1250.0
                    } else {
                        700.0
                    }
                }
                4 => {
                    if g.tech[3] {
                        1230.0
                    } else {
                        700.0
                    }
                }
                9 => {
                    if g.tech[18] {
                        1230.0
                    } else {
                        700.0
                    }
                }
                5 => {
                    if g.tech[7] {
                        1250.0
                    } else {
                        700.0
                    }
                }
                3 => {
                    if g.tech[1] {
                        1180.0
                    } else {
                        700.0
                    }
                }
                11 => {
                    if g.tech[24] {
                        1160.0
                    } else {
                        700.0
                    }
                }
                44 => {
                    if g.tech[44] {
                        1150.0
                    } else {
                        700.0
                    }
                }
                42 => {
                    if g.tech[13] {
                        1140.0
                    } else {
                        700.0
                    }
                }
                2 => 1150.0,
                8 => {
                    if g.tech[17] {
                        1120.0
                    } else {
                        700.0
                    }
                }
                _ => 600.0,
            };
            1100.0 + p - 4.0 * cost as f64
        }
        6 => {
            // build
            let cost = struct_cost(par);
            if g.stars - cost < 2 {
                return -2000.0;
            }
            let p = match par {
                5 => {
                    if ctx.animals <= 1 {
                        700.0
                    } else {
                        320.0
                    }
                }
                6 => 660.0,          // windmill
                3 => 430.0,          // road
                8 => {
                    if sea {
                        920.0
                    } else {
                        260.0
                    }
                }
                48 => {
                    if sea {
                        820.0
                    } else {
                        220.0
                    }
                }
                12 => 420.0, // lumber hut
                21 => 400.0, // mine
                17 => 430.0, // temple
                13 => 360.0, // sawmill
                22 => 320.0, // forge
                23..=29 => -4000.0,
                _ => 250.0,
            };
            700.0 + p - 3.0 * cost as f64
        }
        5 => {
            // harvest: 2 stars for one population, which raises city level (capacity and income)
            if g.stars < 4 {
                return -2000.0;
            }
            1400.0
        }
        3 => {
            match par {
                7 => {
                    // recover
                    let i = match unit_at(g, src as usize) {
                        Some(i) => i,
                        None => return -1e9,
                    };
                    let u = &g.units[i];
                    let maxh = unit_stats(u.kind).0 + if u.veteran { 5.0 } else { 0.0 };
                    if u.health >= maxh - 0.5 {
                        return -1e9;
                    }
                    if g.tiles[u.tile].owner == g.player {
                        430.0
                    } else {
                        120.0
                    }
                }
                16 => 620.0, // promote
                9 => 520.0,  // heal others
                _ => -3000.0,
            }
        }
        1 => {
            // move
            let i = match unit_at(g, src as usize) {
                Some(i) => i,
                None => return -1e9,
            };
            let goal = match goals[i] {
                Some(gl) => gl,
                None => return -2000.0,
            };
            let f = fields.get(goal, is_float(g.units[i].kind));
            let cur = f[g.units[i].tile];
            let d = dst as usize;
            if d >= g.n {
                return -1e9;
            }
            let dv = f[d];
            let cur_c = cheb(g, g.units[i].tile, goal);
            let d_c = cheb(g, d, goal);
            let (progress, better) = if cur < INF && dv < INF {
                ((cur - dv) as f64, dv < cur)
            } else {
                ((cur_c - d_c) as f64, d_c < cur_c)
            };
            if !better {
                return -2000.0;
            }
            300.0 + 40.0 * progress
        }
        10 => -1000.0,
        11 => -1e9,
        _ => -1e9,
    }
}

fn choose(g: &Game, legal: &[Value]) -> String {
    if legal.is_empty() {
        return String::new();
    }
    let ctx = build_ctx(g);
    let (goals, roles) = assign_goals(g, &ctx);
    let fields = build_fields(g, &goals);
    // Is a boat needed? The enemy capital known but unreachable by land.
    let mut sea = false;
    if let Some(ec) = ctx.enemy_capital {
        let f = path_field(g, ec, false);
        sea = f[g.tiles.len() / 2] >= INF && f.iter().filter(|&&d| d < INF).count() < 8;
    }
    let mut best_id = String::new();
    let mut best_score = f64::NEG_INFINITY;
    for a in legal {
        let id = a["action_id"].as_str().unwrap_or("").to_string();
        let s = score_action(g, &ctx, &goals, &roles, &fields, &a["command"], sea);
        if s > best_score {
            best_score = s;
            best_id = id;
        }
    }
    if best_id.is_empty() {
        best_id = legal[0]["action_id"].as_str().unwrap_or("").to_string();
    }
    best_id
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
        let obs: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let g = parse(&obs);
        let legal: Vec<Value> = obs["legal_actions"].as_array().cloned().unwrap_or_default();
        let id = choose(&g, &legal);
        writeln!(stdout, "{}", json!({ "action_id": id })).unwrap();
        stdout.flush().unwrap();
    }
}
