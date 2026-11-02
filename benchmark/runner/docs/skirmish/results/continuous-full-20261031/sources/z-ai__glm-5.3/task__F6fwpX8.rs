// One JSON observation per line; reply {"action_id":"..."} and flush.
// Greedy Bardur 1v1 bot: expansion, economy, tactical combat.
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};

fn gi(v: &Value, k: &str) -> i32 {
    v.get(k).and_then(|x| x.as_i64()).unwrap_or(0) as i32
}
fn gf(v: &Value, k: &str) -> f32 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32
}
fn gb(v: &Value, k: &str) -> bool {
    v.get(k).and_then(|x| x.as_bool()).unwrap_or(false)
}

#[derive(Clone)]
struct Act {
    id: String,
    kind: i32,
    source: i32,
    target: i32,
    parameter: i32,
}

#[derive(Clone)]
struct Unit {
    tile: i32,
    owner: i32,
    kind: i32,
    health: f32,
    veteran: bool,
    moved: bool,
    attacked: bool,
}

// (cost, hp, atk, def, mov, range)
fn stats(kind: i32) -> (i32, f32, f32, f32, i32, i32) {
    match kind {
        2 => (2, 10.0, 2.0, 2.0, 1, 1),
        3 => (3, 10.0, 2.0, 1.0, 2, 1),
        4 => (8, 10.0, 3.5, 1.0, 3, 1),
        5 => (3, 15.0, 1.0, 3.0, 1, 1),
        8 => (8, 10.0, 4.0, 0.0, 1, 3),
        9 => (3, 10.0, 2.0, 1.0, 1, 2),
        10 => (5, 10.0, 0.0, 1.0, 1, 1),
        11 => (5, 15.0, 3.0, 3.0, 1, 1),
        12 => (10, 40.0, 5.0, 4.0, 1, 1),
        15 => (2, 15.0, 3.0, 1.0, 1, 1),
        37 => (2, 10.0, 2.0, 2.0, 1, 1),
        38 => (8, 5.0, 0.0, 0.5, 2, 1),
        39 => (2, 5.0, 0.0, 0.5, 2, 2),
        40 => (2, 10.0, 2.0, 2.0, 2, 1),
        41 => (15, 10.0, 3.0, 2.0, 2, 3),
        42 => (5, 10.0, 2.0, 1.0, 3, 2),
        43 => (0, 10.0, 0.0, 1.0, 2, 2),
        44 => (5, 10.0, 3.0, 3.0, 3, 1),
        45 => (10, 40.0, 4.0, 4.0, 2, 1),
        _ => (3, 10.0, 1.0, 1.0, 1, 1),
    }
}

fn stiff(kind: i32) -> bool {
    matches!(kind, 8 | 10 | 22 | 24 | 34 | 38 | 39 | 41 | 43 | 45 | 56 | 57 | 60 | 61 | 62)
}

fn cheb(a: i32, b: i32, size: i32) -> i32 {
    let dx = (a % size - b % size).abs();
    let dy = (a / size - b / size).abs();
    dx.max(dy)
}

struct CityRow {
    owner: i32,
    level: i32,
}

struct Ctx {
    size: i32,
    turn: i32,
    stars: i32,
    player: i32,
    n: usize,
    known: Vec<bool>,
    terrain: Vec<i32>,
    owner: Vec<i32>,
    structure: Vec<i32>,
    resource: Vec<i32>,
    units: Vec<Unit>,
    acts: Vec<Act>,
    my_city_tiles: Vec<i32>,
    enemy_units: Vec<Unit>,
    my_units: Vec<Unit>,
    water_frac: f32,
    explored_frac: f32,
    cities: Vec<CityRow>,
    my_capital: i32,
}

fn parse(obs: &Value) -> Ctx {
    let vs = &obs["visible_state"];
    let size = gi(vs, "board_size").max(2);
    let n = (size * size) as usize;
    let mut known = vec![false; n];
    let mut terrain = vec![3i32; n];
    let mut owner = vec![0i32; n];
    let mut structure = vec![0i32; n];
    let mut resource = vec![0i32; n];
    if let Some(arr) = vs["tiles"].as_array() {
        for t in arr {
            let i = gi(t, "tile") as usize;
            if i < n {
                known[i] = gb(t, "known");
                if known[i] {
                    terrain[i] = gi(t, "terrain");
                    owner[i] = gi(t, "owner");
                    structure[i] = gi(t, "structure");
                    resource[i] = gi(t, "resource");
                }
            }
        }
    }
    let player = gi(obs, "player").max(1);
    let mut units = Vec::new();
    if let Some(arr) = vs["units"].as_array() {
        for u in arr {
            units.push(Unit {
                tile: gi(u, "tile"),
                owner: gi(u, "owner"),
                kind: gi(u, "kind"),
                health: gf(u, "health"),
                veteran: gb(u, "veteran"),
                moved: gb(u, "moved"),
                attacked: gb(u, "attacked"),
            });
        }
    }
    let mut acts = Vec::new();
    if let Some(arr) = obs["legal_actions"].as_array() {
        for a in arr {
            let c = &a["command"];
            acts.push(Act {
                id: a["action_id"].as_str().unwrap_or("").to_string(),
                kind: gi(c, "kind"),
                source: gi(c, "source"),
                target: gi(c, "target"),
                parameter: gi(c, "parameter"),
            });
        }
    }
    let mut my_capital = -1i32;
    if let Some(arr) = vs["tiles"].as_array() {
        for t in arr {
            if gb(t, "known") && gi(t, "capital") == player && gi(t, "owner") == player {
                my_capital = gi(t, "tile");
            }
        }
    }
    let mut cities = Vec::new();
    if let Some(arr) = vs["cities"].as_array() {
        for c in arr {
            cities.push(CityRow { owner: gi(c, "owner"), level: gi(c, "level") });
        }
    }
    let mut my_city_tiles = Vec::new();
    for t in 0..n {
        if structure[t] == 1 && owner[t] == player {
            my_city_tiles.push(t as i32);
        }
    }
    let enemy_units: Vec<Unit> = units.iter().filter(|u| u.owner != player && u.owner > 0).cloned().collect();
    let my_units: Vec<Unit> = units.iter().filter(|u| u.owner == player).cloned().collect();
    let known_count = known.iter().filter(|&&k| k).count().max(1);
    let water = (0..n).filter(|&t| known[t] && terrain[t] <= 2).count();
    Ctx {
        size,
        turn: gi(vs, "turn"),
        stars: gi(vs, "stars"),
        player,
        n,
        known,
        terrain,
        owner,
        structure,
        resource,
        units,
        acts,
        my_city_tiles,
        enemy_units,
        my_units,
        water_frac: water as f32 / known_count as f32,
        explored_frac: known_count as f32 / n as f32,
        cities,
        my_capital,
    }
}

fn unit_at(ctx: &Ctx, tile: i32) -> Option<&Unit> {
    ctx.units.iter().find(|u| u.tile == tile && u.owner > 0)
}

fn max_hp(u: &Unit) -> f32 {
    stats(u.kind).1 + if u.veteran { 5.0 } else { 0.0 }
}

struct Mem {
    turn: i32,
    claims: HashSet<i32>,
    assign: HashMap<i32, i32>,
    gar: i32,
}

fn choose(ctx: &Ctx, mem: &mut Mem) -> String {
    if ctx.turn != mem.turn {
        mem.turn = ctx.turn;
        mem.claims.clear();
    }
    let enemy = if ctx.player == 1 { 2 } else { 1 };
    let size = ctx.size;

    // ---- 1. mandatory city rewards
    let war_early = !ctx.enemy_units.is_empty() || ctx.turn < 22;
    let mut best_reward: Option<&Act> = None;
    let mut best_rs = -1.0f32;
    for a in &ctx.acts {
        if a.kind == 9 {
            let wall_ok = war_early || a.target == ctx.my_capital;
            let s = match a.parameter {
                3 => 55.0,
                7 => 52.0,
                8 => 48.0,
                6 => 46.0,
                4 => 40.0,
                5 => 34.0,
                1 => { if wall_ok { 58.0 } else { 30.0 } }
                2 => 22.0,
                _ => 0.0,
            };
            if s > best_rs {
                best_rs = s;
                best_reward = Some(a);
            }
        }
    }
    if let Some(a) = best_reward {
        return a.id.clone();
    }

    // ---- 2. captures (enemy city > ruin > neutral village)
    let mut best_cap: Option<&Act> = None;
    let mut best_cs = -1i32;
    for a in &ctx.acts {
        if a.kind == 8 {
            let t = a.source as usize;
            let s = if t < ctx.n && ctx.structure[t] == 1 && ctx.owner[t] == enemy {
                3
            } else if t < ctx.n && ctx.structure[t] == 2 {
                2
            } else {
                1
            };
            if s > best_cs {
                best_cs = s;
                best_cap = Some(a);
            }
        }
    }
    if let Some(a) = best_cap {
        mem.assign.remove(&a.source);
        mem.claims.clear();
        return a.id.clone();
    }

    // ---- 3. promote (free)
    for a in &ctx.acts {
        if a.kind == 3 && a.parameter == 16 {
            return a.id.clone();
        }
    }

    // ---- 4. attacks
    let my_city_tiles_ref = ctx.my_city_tiles.clone();
    let mut best_atk: Option<&Act> = None;
    let mut best_as = 3.0f32;
    for a in &ctx.acts {
        if a.kind != 2 {
            continue;
        }
        let au = match unit_at(ctx, a.source) {
            Some(u) if u.owner == ctx.player => u,
            _ => continue,
        };
        let du = match unit_at(ctx, a.target) {
            Some(u) if u.owner == enemy => u,
            _ => continue,
        };
        let (_, _, atk_a, _, _, _) = stats(au.kind);
        let (cost_d, _, atk_d, def_d, _, rng_d) = stats(du.kind);
        let af = atk_a * (au.health / max_hp(au).max(1.0));
        let dt = a.target as usize;
        let mut bonus = 1.0f32;
        if dt < ctx.n && ctx.structure[dt] == 1 && ctx.owner[dt] == du.owner {
            bonus = 1.8;
        } else if dt < ctx.n && (ctx.terrain[dt] == 4 || ctx.terrain[dt] == 5) {
            bonus = 1.3;
        }
        let df = def_d * (du.health / max_hp(du).max(1.0)) * bonus;
        let total = af + df;
        if total <= 0.0 {
            continue;
        }
        let dmg = ((af / total) * atk_a * 4.5 + 0.5).floor();
        let kill = dmg >= du.health;
        let mut ret = 0.0f32;
        if !kill && atk_d > 0.0 && !stiff(du.kind) && cheb(a.source, a.target, size) <= rng_d {
            ret = ((df / total) * def_d * 4.5 + 0.5).floor();
        }
        let mut value = dmg - 0.8 * ret;
        if kill {
            value += 25.0 + 4.0 * cost_d as f32;
        } else if ret >= au.health {
            value -= 18.0 + 4.0 * stats(au.kind).0 as f32;
        }
        if dt < ctx.n && ctx.structure[dt] == 1 && ctx.owner[dt] == ctx.player {
            value += 25.0; // relieve siege
        }
        if my_city_tiles_ref.iter().any(|&c| cheb(a.target, c, size) <= 3) {
            value += 10.0; // defending home ground
        }
        if value > best_as {
            best_as = value;
            best_atk = Some(a);
        }
    }
    if let Some(a) = best_atk {
        return a.id.clone();
    }

    // ---- shared context facts
    let war = !ctx.enemy_units.is_empty();
    let mut enemy_near_city = false;
    for e in &ctx.enemy_units {
        for &c in &ctx.my_city_tiles {
            if cheb(e.tile, c, size) <= 7 {
                enemy_near_city = true;
            }
        }
    }

    // ---- 5. movement
    {
        let mut per_src: HashMap<i32, Vec<&Act>> = HashMap::new();
        for a in &ctx.acts {
            if a.kind == 1 {
                per_src.entry(a.source).or_default().push(a);
            }
        }
        // target validity + base score
        let cap0 = if ctx.my_capital >= 0 { ctx.my_capital } else if !ctx.my_city_tiles.is_empty() { ctx.my_city_tiles[0] } else { 0 };
        let strong = ctx.my_units.len() >= 5 || war;
        let valid_target = |t: i32| -> Option<i32> {
            let tu = t as usize;
            if tu >= ctx.n || !ctx.known[tu] {
                return None;
            }
            if ctx.structure[tu] == 1 && ctx.owner[tu] == 0 {
                return Some(60);
            }
            if ctx.structure[tu] == 2 {
                return Some(52);
            }
            if ctx.structure[tu] == 1 && ctx.owner[tu] == enemy {
                return Some(if strong { 62 } else { 12 });
            }
            // frontier: known tile with unknown neighbours
            let tx = t % size;
            let ty = t / size;
            let mut unk = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nx = tx + dx;
                    let ny = ty + dy;
                    if nx >= 0 && nx < size && ny >= 0 && ny < size && !ctx.known[(ny * size + nx) as usize] {
                        unk += 1;
                    }
                }
            }
            if unk > 0 && ctx.owner[tu] != ctx.player {
                let tx = t % size;
                let ty = t / size;
                let edge = tx.min(size - 1 - tx).min(ty.min(size - 1 - ty)).min(5);
                return Some(2 + 4 * unk + 2 * edge + (cheb(cap0, t, size) / 3).min(3));
            }
            None
        };
        let mut neutral_villages = 0i32;
        for t in 0..ctx.n {
            if ctx.known[t] && ctx.structure[t] == 1 && ctx.owner[t] == 0 {
                neutral_villages += 1;
            }
        }
        let _ = neutral_villages;
        // candidate tile list (for fresh assignment)
        let mut cands: Vec<i32> = Vec::new();
        for t in 0..ctx.n {
            let ti = t as i32;
            if valid_target(ti).is_some() {
                cands.push(ti);
            }
        }
        if enemy_near_city {
            for e in &ctx.enemy_units {
                cands.push(e.tile);
            }
        }
        if ctx.my_units.len() >= 2 && ctx.turn > 4 {
            for &c in &ctx.my_city_tiles {
                let guarded = ctx.my_units.iter().any(|u| cheb(u.tile, c, size) <= 2);
                if !guarded {
                    cands.push(c);
                }
            }
        }

        let mut best_move: Option<(f32, &Act, i32)> = None;
        // garrison: keep one unit parked on the capital while vulnerable
        let cap = if ctx.my_capital >= 0 { ctx.my_capital } else if !ctx.my_city_tiles.is_empty() { ctx.my_city_tiles[0] } else { -1 };
        let garrison_needed = cap >= 0
            && ((ctx.my_city_tiles.len() <= 2 && ctx.turn >= 3) || ctx.enemy_units.iter().any(|e| cheb(e.tile, cap, size) <= 9));
        let mut gar_unit: i32 = -1;
        if garrison_needed {
            let sticky = mem.gar;
            let sticky_ok = sticky >= 0 && ctx.my_units.iter().any(|u| u.tile == sticky);
            gar_unit = if sticky_ok {
                sticky
            } else {
                let pool: Vec<&Unit> = ctx.my_units.iter().filter(|u| !u.moved).collect();
                match pool.iter().find(|u| cheb(u.tile, cap, size) <= 2) {
                    Some(u) => u.tile,
                    None => match pool.iter().min_by_key(|u| cheb(u.tile, cap, size)) {
                        Some(u) => u.tile,
                        None => -1,
                    },
                }
            };
            mem.gar = gar_unit;
        } else {
            mem.gar = -1;
        }
        for u in &ctx.my_units {
            if u.moved {
                continue;
            }
            let mv = match per_src.get(&u.tile) {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            };
            let atk_u = stats(u.kind).2;
            if gar_unit >= 0 && u.tile == gar_unit && u.tile != cap {
                // walk the garrison home
                let mut local_best: Option<(f32, &Act)> = None;
                for a in mv {
                    let mut s = -10.0 * cheb(a.target, cap, size) as f32;
                    if a.target == cap {
                        s += 200.0;
                    }
                    if s > local_best.map(|(x, _)| x).unwrap_or(-100.0) {
                        local_best = Some((s, a));
                    }
                }
                if let Some((_, a)) = local_best {
                    mem.assign.remove(&a.source);
                    mem.assign.insert(a.target, cap);
                    mem.gar = a.target;
                    return a.id.clone();
                }
                continue;
            }
            if u.tile == cap && garrison_needed {
                continue; // garrison stays put
            }
            // persistent target (ignored when defending home)
            let mut target: i32 = 0;
            let mut have_target = false;
            if !enemy_near_city {
                if let Some(&t) = mem.assign.get(&u.tile) {
                    if valid_target(t).is_some() {
                        target = t;
                        have_target = true;
                    }
                }
            }
            if !have_target {
                let mut bt = (0i32, i32::MIN);
                for &t in &cands {
                    if mem.claims.contains(&t) {
                        continue;
                    }
                    let mut b = match valid_target(t) {
                        Some(b) => b,
                        None => {
                            // enemy unit or undefended city target
                            if ctx.enemy_units.iter().any(|e| e.tile == t) {
                                70
                            } else {
                                55
                            }
                        }
                    };
                    if atk_u <= 0.0 {
                        let tu = t as usize;
                        if tu < ctx.n && ctx.structure[tu] == 1 && ctx.owner[tu] == enemy {
                            b /= 4;
                        }
                    }
                    let pen = if b <= 40 { 2 } else { 3 };
                    let s = b - pen * cheb(u.tile, t, size);
                    if s > bt.1 {
                        bt = (t, s);
                    }
                }
                if bt.1 == i32::MIN {
                    continue;
                }
                target = bt.0;
                mem.claims.insert(target);
            }
            // sole defender of a threatened city must stay close
            let mut hold: Option<i32> = None;
            for &c in &ctx.my_city_tiles {
                let near = ctx.enemy_units.iter().any(|e| cheb(e.tile, c, size) <= 6);
                if near && cheb(u.tile, c, size) <= 1 {
                    let others = ctx.my_units.iter().any(|o| o.tile != u.tile && cheb(o.tile, c, size) <= 2);
                    if !others {
                        hold = Some(c);
                    }
                }
            }
            let stay = -10.0 * cheb(u.tile, target, size) as f32;
            let mut local_best: Option<(f32, &Act)> = None;
            for a in mv {
                let t = a.target;
                if let Some(c) = hold {
                    if cheb(t, c, size) > 1 {
                        continue;
                    }
                }
                let mut s = -10.0 * cheb(t, target, size) as f32;
                let tu = t as usize;
                if t == target {
                    s += 100.0;
                }
                if tu < ctx.n && ((ctx.structure[tu] == 1 && ctx.owner[tu] != ctx.player) || ctx.structure[tu] == 2) {
                    s += 55.0;
                }
                for e in &ctx.enemy_units {
                    let (_, _, _, _, mov_e, rng_e) = stats(e.kind);
                    if cheb(t, e.tile, size) <= mov_e + rng_e {
                        s -= if u.health < 6.0 { 25.0 } else { 10.0 };
                    }
                }
                if tu < ctx.n {
                    if ctx.terrain[tu] == 4 {
                        s += 3.0;
                    } else if ctx.terrain[tu] == 5 {
                        s += 2.0;
                    }
                }
                if s > stay + 0.5 && (local_best.is_none() || s > local_best.as_ref().unwrap().0) {
                    local_best = Some((s, a));
                }
            }
            if let Some((s, a)) = local_best {
                if best_move.is_none() || s > best_move.as_ref().unwrap().0 {
                    best_move = Some((s, a, target));
                }
            }
        }
        if let Some((_, a, tgt)) = best_move {
            mem.assign.remove(&a.source);
            mem.assign.insert(a.target, tgt);
            return a.id.clone();
        }
    }

    // ---- 6. train    // ---- 6. train units (up to city capacity)
    {
        let mut capacity = 0i32;
        for c in &ctx.cities {
            if c.owner == ctx.player {
                capacity += c.level;
            }
        }
        let have = ctx.my_units.len() as i32;
        let pref: &[i32] = if war { &[11, 5, 9, 2, 3, 4] } else { &[3, 2, 9, 5, 11, 4] };
        let mut chosen: Option<&Act> = None;
        let mut best_rank = 99usize;
        for a in &ctx.acts {
            if a.kind == 4 {
                if let Some(r) = pref.iter().position(|&k| k == a.parameter) {
                    if r < best_rank {
                        best_rank = r;
                        chosen = Some(a);
                    }
                }
            }
        }
        if let Some(a) = chosen {
            let cost = stats(a.parameter).0;
            let reserve = if war { 1 } else if ctx.my_city_tiles.len() < 4 { 0 } else { 3 };
            if have < capacity && ctx.stars - cost >= reserve {
                return a.id.clone();
            }
        }
    }

    // ---- 7. harvest resources
    if ctx.stars >= 2 {
        if let Some(a) = ctx.acts.iter().find(|a| a.kind == 5) {
            return a.id.clone();
        }
    }

    // ---- 7b. research
    {
        let mut order: Vec<i32> = vec![6, 16, 1, 20];
        if ctx.water_frac > 0.12 {
            order.push(10);
        }
        order.push(18);
        order.push(7);
        order.push(23);
        order.push(24);
        if ctx.water_frac > 0.35 {
            order.push(13);
        }
        order.push(8);
        order.push(4);
        order.push(5);
        order.push(9);
        order.push(17);
        order.push(3);
        for t in order {
            if let Some(a) = ctx.acts.iter().find(|a| a.kind == 7 && a.parameter == t) {
                let tier = match t { 1 | 6 | 20 | 10 => 1, 13 => 2, 3 | 5 | 9 | 17 | 22 => 3, _ => 2 };
                let cities = ctx.my_city_tiles.len() as i32;
                let cost = 4 + cities * tier;
                let gate = cost + if tier == 1 { 0 } else { 2 };
                if ctx.stars >= gate {
                    return a.id.clone();
                }
            }
        }
    }

    // ---- 9. build
    {
        let mut neutral_villages = 0i32;
        for t in 0..ctx.n {
            if ctx.known[t] && ctx.structure[t] == 1 && ctx.owner[t] == 0 {
                neutral_villages += 1;
            }
        }
        let naval = neutral_villages == 0 && ctx.turn > 6;
        let mut best_b: Option<&Act> = None;
        let mut best_bs = -1.0f32;
        for a in &ctx.acts {
            if a.kind != 6 {
                continue;
            }
            let (s, cost) = match a.parameter {
                8 => (if naval || ctx.water_frac > 0.45 { 65.0 } else { 15.0 }, 7),
                5 => (55.0, 5),
                21 => (50.0, 5),
                12 => (52.0, 3),
                6 => (25.0, 5),
                13 => (25.0, 5),
                22 => (25.0, 5),
                3 => (18.0, 3),
                _ => (8.0, 5),
            };
            if ctx.stars - cost >= 2 && s > best_bs {
                best_bs = s;
                best_b = Some(a);
            }
        }
        if let Some(a) = best_b {
            return a.id.clone();
        }
    }

    // ---- 10. recover hurt idle units (safe ones)
    for a in &ctx.acts {
        if a.kind == 3 && a.parameter == 7 {
            if let Some(u) = unit_at(ctx, a.source) {
                if u.owner == ctx.player && u.health <= max_hp(u) - 5.0 {
                    let threat = ctx.enemy_units.iter().any(|e| cheb(e.tile, u.tile, size) <= 2);
                    if !threat {
                        return a.id.clone();
                    }
                }
            }
        }
    }

    // ---- 11. end turn
    if let Some(a) = ctx.acts.iter().find(|a| a.kind == 10) {
        return a.id.clone();
    }
    // fallback: anything but resign
    if let Some(a) = ctx.acts.iter().find(|a| a.kind != 11) {
        return a.id.clone();
    }
    ctx.acts.first().map(|a| a.id.clone()).unwrap_or_else(|| "0".to_string())
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut mem = Mem { turn: -1, claims: HashSet::new(), assign: HashMap::new(), gar: -1 };
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
        let id = choose(&parse(&obs), &mut mem);
        let _ = writeln!(out, "{}", json!({"action_id": id}));
        let _ = out.flush();
    }
}
