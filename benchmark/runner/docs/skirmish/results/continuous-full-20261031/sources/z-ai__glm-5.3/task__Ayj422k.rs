// Polytopia Bardur 1v1 heuristic bot.
// One JSON observation per line; reply {"action_id":"..."} and flush.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

#[derive(Clone)]
struct Tile {
    known: bool,
    terrain: i64,
    owner: i64,
    structure: i64,
    resource: i64,
}

struct U {
    tile: usize,
    kind: i64,
    hp: f32,
    owner: i64,
    moved: bool,
    attacked: bool,
    vet: bool,
}

// (cost, maxhp, atk, def, mov, range)
fn stats(kind: i64) -> (i32, f32, f32, f32, i32, i32) {
    match kind {
        2 => (2, 10.0, 2.0, 2.0, 1, 1),      // Warrior
        3 => (3, 10.0, 2.0, 1.0, 2, 1),      // Rider
        4 => (8, 10.0, 3.5, 1.0, 3, 1),      // Knight
        5 => (3, 15.0, 1.0, 3.0, 1, 1),      // Defender
        8 => (8, 10.0, 4.0, 0.0, 1, 3),      // Catapult
        9 => (3, 10.0, 2.0, 1.0, 1, 2),      // Archer
        10 => (5, 10.0, 0.0, 1.0, 1, 1),     // MindBender
        11 => (5, 15.0, 3.0, 3.0, 1, 1),     // Swordsman
        12 => (10, 40.0, 5.0, 4.0, 1, 1),    // Giant
        15 => (2, 15.0, 3.0, 1.0, 1, 1),     // Polytaur
        37 => (2, 10.0, 2.0, 2.0, 1, 1),     // Dagger
        38 => (8, 5.0, 0.0, 0.5, 2, 1),      // Cloak
        39 => (2, 5.0, 0.0, 0.5, 2, 1),      // Dinghy
        40 => (2, 10.0, 2.0, 2.0, 2, 1),     // Pirate
        41 => (15, 10.0, 3.0, 2.0, 2, 3),    // Bomber
        42 => (5, 10.0, 2.0, 1.0, 3, 2),     // Scoutship
        43 => (0, 10.0, 0.0, 1.0, 2, 2),     // Raft
        44 => (5, 10.0, 3.0, 3.0, 3, 1),     // Rammer
        45 => (10, 40.0, 4.0, 4.0, 2, 1),    // Juggernaut
        _ => (3, 10.0, 2.0, 1.0, 1, 1),
    }
}

fn is_stiff(kind: i64) -> bool {
    matches!(kind, 8 | 10 | 38 | 39 | 41 | 43 | 45)
}

fn cheb(a: usize, b: usize, size: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

fn def_bonus(t: &Tile) -> f32 {
    if t.structure == 1 {
        1.5
    } else if t.terrain == 4 {
        1.5
    } else if t.terrain == 5 {
        1.3
    } else {
        1.0
    }
}

/// Expected damage dealt by attacker onto defender.
fn damage(a_atk: f32, a_hp: f32, a_max: f32, d_def: f32, d_hp: f32, d_max: f32, bonus: f32) -> f32 {
    let af = a_atk * (a_hp / a_max.max(1.0));
    let df = d_def * (d_hp / d_max.max(1.0)) * bonus;
    let tot = af + df;
    if tot <= 0.0 {
        return 0.0;
    }
    ((af / tot) * a_atk * 4.5 + 0.5).floor()
}

/// BFS distance field from sources; enter = tiles that may be stepped onto,
/// expand = tiles the search may continue from. Unknown/unreached -> large.
fn bfs(n: usize, size: usize, sources: &[usize], enter: &[bool], expand: &[bool]) -> Vec<i32> {
    let mut dist = vec![i32::MAX / 4; n];
    let mut queue = std::collections::VecDeque::new();
    for &s in sources {
        if s < n {
            dist[s] = 0;
            queue.push_back(s);
        }
    }
    while let Some(x) = queue.pop_front() {
        let d = dist[x];
        if !expand[x] {
            continue;
        }
        let (xx, yy) = ((x % size) as i32, (x / size) as i32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (xx + dx, yy + dy);
                if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 {
                    continue;
                }
                let j = (ny as usize) * size + nx as usize;
                if enter[j] && dist[j] > d + 1 {
                    dist[j] = d + 1;
                    queue.push_back(j);
                }
            }
        }
    }
    dist
}

struct Bot {
    claims_turn: i64,
    last_turn: i64,
    last_version: i64,
    claimed: std::collections::HashSet<usize>,
}

impl Bot {
    fn new() -> Bot {
        Bot { claims_turn: -1, last_turn: -1, last_version: -1, claimed: std::collections::HashSet::new() }
    }

    fn choose(&mut self, obs: &Value) -> String {
        let v = &obs["visible_state"];
        let size = v["board_size"].as_i64().unwrap_or(14) as usize;
        let n = size * size;
        let me = obs["player"].as_i64().unwrap_or(1);
        let enemy_seat = if me == 1 { 2 } else { 1 };
        let turn = v["turn"].as_i64().unwrap_or(0);
        let stars = v["stars"].as_i64().unwrap_or(0);
        let version = obs["state_version"].as_i64().unwrap_or(0);

        if turn < self.last_turn || version < self.last_version {
            self.claims_turn = -1;
            self.claimed.clear();
        }
        self.last_turn = turn;
        self.last_version = version;
        if self.claims_turn != turn {
            self.claims_turn = turn;
        }

        // ---- parse tiles
        let mut tiles = vec![
            Tile { known: false, terrain: 0, owner: 0, structure: 0, resource: 0 };
            n
        ];
        if let Some(arr) = v["tiles"].as_array() {
            for t in arr {
                let i = t["tile"].as_i64().unwrap_or(0) as usize;
                if i >= n {
                    continue;
                }
                let known = t["known"].as_bool().unwrap_or(false);
                tiles[i] = Tile {
                    known,
                    terrain: t["terrain"].as_i64().unwrap_or(3),
                    owner: t["owner"].as_i64().unwrap_or(0),
                    structure: t["structure"].as_i64().unwrap_or(0),
                    resource: t["resource"].as_i64().unwrap_or(0),
                };
            }
        }

        // ---- tech
        let mut tech = vec![false; 128];
        if let Some(arr) = v["technology"].as_array() {
            for t in arr {
                if let Some(k) = t.as_i64() {
                    if k >= 0 && k < 128 {
                        tech[k as usize] = true;
                    }
                }
            }
        }

        // ---- units
        let mut units: Vec<U> = Vec::new();
        if let Some(arr) = v["units"].as_array() {
            for u in arr {
                units.push(U {
                    tile: u["tile"].as_i64().unwrap_or(0) as usize,
                    kind: u["kind"].as_i64().unwrap_or(2),
                    hp: u["health"].as_f64().unwrap_or(10.0) as f32,
                    owner: u["owner"].as_i64().unwrap_or(0),
                    moved: u["moved"].as_bool().unwrap_or(true),
                    attacked: u["attacked"].as_bool().unwrap_or(true),
                    vet: u["veteran"].as_bool().unwrap_or(false),
                });
            }
        }
        let unit_at = |tile: usize| -> Option<&U> { units.iter().find(|u| u.tile == tile) };

        // ---- own cities (tile -> level)
        let mut my_city_level: HashMap<usize, i64> = HashMap::new();
        if let Some(arr) = v["cities"].as_array() {
            for c in arr {
                if c["owner"].as_i64().unwrap_or(0) == me {
                    my_city_level.insert(c["tile"].as_i64().unwrap_or(0) as usize, c["level"].as_i64().unwrap_or(1));
                }
            }
        }

        // ---- legal actions
        let mut moves: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut move_ids: HashMap<(usize, usize), String> = HashMap::new();
        let mut attacks: Vec<(usize, usize, String)> = Vec::new();
        let mut captures: Vec<(usize, String)> = Vec::new();
        let mut summons: Vec<(usize, i64, String)> = Vec::new();
        let mut builds: Vec<(usize, i64, String)> = Vec::new();
        let mut harvests: Vec<(usize, String)> = Vec::new();
        let mut researches: Vec<(i64, String)> = Vec::new();
        let mut rewards: Vec<(usize, i64, String)> = Vec::new();
        let mut promotes: Vec<String> = Vec::new();
        let mut end_id: Option<String> = None;
        let mut first_id: Option<String> = None;

        if let Some(arr) = obs["legal_actions"].as_array() {
            for a in arr {
                let id = a["action_id"].as_str().unwrap_or("").to_string();
                if first_id.is_none() {
                    first_id = Some(id.clone());
                }
                let c = &a["command"];
                let kind = c["kind"].as_i64().unwrap_or(0);
                let src = c["source"].as_i64().unwrap_or(-1);
                let dst = c["target"].as_i64().unwrap_or(-1);
                let par = c["parameter"].as_i64().unwrap_or(0);
                match kind {
                    1 => {
                        if src >= 0 && dst >= 0 {
                            moves.entry(src as usize).or_default().push(dst as usize);
                            move_ids.insert((src as usize, dst as usize), id.clone());
                        }
                    }
                    2 => attacks.push((src as usize, dst as usize, id)),
                    4 => summons.push((src as usize, par, id)),
                    5 => harvests.push((dst as usize, id)),
                    6 => builds.push((dst as usize, par, id)),
                    7 => researches.push((par, id)),
                    8 => captures.push((src as usize, id)),
                    9 => rewards.push((dst as usize, par, id)),
                    3 if par == 16 => promotes.push(id),
                    10 => end_id = Some(id),
                    _ => {}
                }
            }
        }

        // ============ 1. mandatory city rewards ============
        if !rewards.is_empty() {
            let pref = [3, 7, 8, 6, 4, 1, 5, 2];
            for &p in pref.iter() {
                if let Some((_, _, id)) = rewards.iter().find(|(_, par, _)| *par == p) {
                    return id.clone();
                }
            }
            return rewards[0].2.clone();
        }

        // ============ 2. promote veterans (free) ============
        if let Some(id) = promotes.first() {
            return id.clone();
        }

        // ============ 3. captures ============
        if !captures.is_empty() {
            let prio = |t: &Tile| -> i32 {
                if t.structure == 1 && t.owner == enemy_seat {
                    0
                } else if t.structure == 1 && t.owner == 0 {
                    1
                } else if t.resource == 8 {
                    2
                } else if t.structure == 2 {
                    3
                } else {
                    4
                }
            };
            let mut best: Option<(i32, String)> = None;
            for (src, id) in &captures {
                let p = if *src < n { prio(&tiles[*src]) } else { 4 };
                if best.as_ref().map_or(true, |(bp, _)| p < *bp) {
                    best = Some((p, id.clone()));
                }
            }
            if let Some((_, id)) = best {
                return id;
            }
        }

        // ============ 4. attacks ============
        {
            let my_units_total0 = units.iter().filter(|u| u.owner == me).count();
            let enemy_units_seen0 = units.iter().filter(|u| u.owner == enemy_seat).count();
            let mut best: Option<(f32, String)> = None;
            for (src, dst, id) in &attacks {
                let au = match unit_at(*src) {
                    Some(u) => u,
                    None => continue,
                };
                let du = match unit_at(*dst) {
                    Some(u) if u.owner == enemy_seat => u,
                    _ => continue,
                };
                let (_, a_max, a_atk, _, _, _) = stats(au.kind);
                let a_max = a_max + if au.vet { 5.0 } else { 0.0 };
                let (_, d_max, _, d_def, _, d_rng) = stats(du.kind);
                let d_max = d_max + if du.vet { 5.0 } else { 0.0 };
                let bonus = if *dst < n { def_bonus(&tiles[*dst]) } else { 1.0 };
                let d = damage(a_atk, au.hp, a_max, d_def, du.hp, d_max, bonus);
                let kill = d >= du.hp;
                let dist = cheb(*src, *dst, size);
                let mut ret = 0.0;
                if !kill && !is_stiff(du.kind) && d_rng >= dist {
                    // retaliation: defender attacks back (its attack stat)
                    let (_, _, d_atk, _, _, _) = stats(du.kind);
                    let (_, _, _, a_def, _, _) = stats(au.kind);
                    let src_bonus = if *src < n { def_bonus(&tiles[*src]) } else { 1.0 };
                    ret = damage(d_atk, du.hp - d, d_max, a_def, au.hp, a_max, src_bonus);
                }
                let we_die = ret >= au.hp;
                let mut score = d + if kill { 25.0 } else { 0.0 } - 1.3 * ret - if we_die { 30.0 } else { 0.0 };
                if *dst < n && tiles[*dst].owner == me {
                    score += 8.0; // defending our land
                }
                if *dst < n && tiles[*dst].structure == 1 && tiles[*dst].owner == enemy_seat {
                    score += 10.0;
                }
                if kill && !we_die {
                    score += 5.0;
                }
                if best.as_ref().map_or(true, |(bs, _)| score > *bs) {
                    best = Some((score, id.clone()));
                }
            }
            if let Some((s, id)) = best {
                let gate = if enemy_units_seen0 > my_units_total0 { 9.0 } else { 1.0 };
                if s > gate {
                    return id;
                }
            }
        }

        // ============ context ============
        let known_count = tiles.iter().filter(|t| t.known).count();
        let water_known = tiles.iter().filter(|t| t.known && t.terrain <= 2).count();
        let water_frac = if known_count > 0 { water_known as f32 / known_count as f32 } else { 0.0 };
        let enemy_city_known = tiles.iter().any(|t| t.known && t.structure == 1 && t.owner == enemy_seat);
        let enemy_unit_seen = units.iter().any(|u| u.owner == enemy_seat);
        let my_units_total = units.iter().filter(|u| u.owner == me).count();
        let enemy_units_seen = units.iter().filter(|u| u.owner == enemy_seat).count();
        let outnumbered = enemy_units_seen > 0 && my_units_total + 3 < enemy_units_seen;

        // unclaimed neutral villages / ruins / enemy cities
        let mut villages: Vec<usize> = (0..n)
            .filter(|&i| tiles[i].known && tiles[i].structure == 1 && tiles[i].owner == 0 && !self.claimed.contains(&i))
            .collect();
        let ruins: Vec<usize> = (0..n).filter(|&i| tiles[i].known && tiles[i].structure == 2).collect();
        let enemy_cities: Vec<usize> = (0..n)
            .filter(|&i| tiles[i].known && tiles[i].structure == 1 && tiles[i].owner == enemy_seat)
            .collect();
        let my_cities_list: Vec<usize> = (0..n)
            .filter(|&i| tiles[i].known && tiles[i].structure == 1 && tiles[i].owner == me)
            .collect();
        let threat: Vec<usize> = units
            .iter()
            .filter(|u| u.owner == enemy_seat)
            .map(|u| u.tile)
            .filter(|&t| my_cities_list.iter().any(|&c| cheb(t, c, size) <= 5))
            .collect();
        let threatened_cities: Vec<usize> = my_cities_list
            .iter()
            .copied()
            .filter(|&c| threat.iter().any(|&t| cheb(t, c, size) <= 5))
            .collect();
        let mut garrisons: Vec<usize> = threatened_cities
            .iter()
            .copied()
            .filter(|&c| !units.iter().any(|u| u.owner == me && u.tile == c))
            .collect();
        garrisons.sort_by_key(|&c| {
            threat.iter().map(|&t| cheb(t, c, size)).min().unwrap_or(99)
        });
        garrisons.truncate(2);

        // ============ 5. research ============
        if !researches.is_empty() {
            let prio: Vec<i64> = if water_frac > 0.15 {
                vec![6, 1, 10, 13, 16, 18, 8, 20, 23, 7, 9, 24, 17, 3, 21, 22, 44, 14]
            } else {
                vec![6, 1, 16, 18, 10, 8, 13, 23, 20, 7, 9, 24, 17, 3, 21, 22, 14]
            };
            let tier = |k: i64| -> i64 {
                match k {
                    1 | 6 | 10 | 20 | 25 => 1,
                    2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
                    _ => 3,
                }
            };
            let city_count = my_cities_list.len() as i64;
            for &p in prio.iter() {
                if tech[p as usize] {
                    continue;
                }
                if let Some((_, id)) = researches.iter().find(|(par, _)| *par == p) {
                    let cost = 4 + city_count * tier(p);
                    if stars >= cost + 6 || turn > 40 || stars > 120 {
                        return id.clone();
                    }
                }
                // missing or not yet affordable: fall through to next priority
            }
        }

        // ============ 6. builds ============
        if !builds.is_empty() {
            let build_cost = |par: i64| -> i64 {
                match par {
                    12 => 3,  // lumber hut
                    3 => 3,   // road
                    5 => 5,   // farm
                    21 => 5,  // mine
                    13 => 5,  // sawmill
                    6 => 5,   // windmill
                    22 => 5,  // forge
                    8 => 7,   // port
                    48 => 5,  // bridge
                    50 => 5,  // market
                    32 => 5,  // sanctuary
                    _ => 20,  // temples etc
                }
            };
            // naval need: water map and enemy not reachable / late
            let naval = water_frac > 0.28 && tech[10] && (enemy_city_known || turn > 15);
            let mut pref: Vec<i64> = if naval {
                vec![8, 12, 21, 5, 13, 22, 6, 50, 3]
            } else {
                vec![12, 21, 5, 13, 22, 6, 50, 3, 8]
            };
            if stars >= 40 {
                pref.extend_from_slice(&[18, 17, 20, 19]);
            }
            for &p in pref.iter() {
                let cands: Vec<&(usize, i64, String)> = builds.iter().filter(|(_, par, _)| *par == p).collect();
                if cands.is_empty() {
                    continue;
                }
                let cost = build_cost(p);
                if stars >= cost + 4 || turn > 45 {
                    // pick candidate: for ports prefer near enemy; else first
                    if p == 8 {
                        let mut best: Option<(i32, String)> = None;
                        for (t, _, id) in &cands {
                            let mut d = i32::MAX;
                            for &ec in &enemy_cities {
                                d = d.min(cheb(*t, ec, size));
                            }
                            if d == i32::MAX {
                                d = 20;
                            }
                            if best.as_ref().map_or(true, |(bd, _)| d < *bd) {
                                best = Some((d, id.clone()));
                            }
                        }
                        if let Some((_, id)) = best {
                            return id;
                        }
                    }
                    return cands[0].2.clone();
                }
                break;
            }
        }

        // ============ 7. harvest ============
        if !harvests.is_empty() && stars >= 4 {
            return harvests[0].1.clone();
        }

        // ============ 8. summon (train units) ============
        if !summons.is_empty() {
            let want_army = enemy_unit_seen || enemy_city_known || turn >= 20;
            let combat = !threat.is_empty() || enemy_city_known || turn >= 20;
            let expanding = !villages.is_empty() && my_units_total < 16;
            let kind_pref: Vec<i64> = if outnumbered {
                vec![5, 11, 3, 2, 9]
            } else if want_army {
                vec![11, 3, 2, 9, 5]
            } else {
                vec![3, 2]
            };
            for &k in kind_pref.iter() {
                let cands: Vec<&(usize, i64, String)> = summons.iter().filter(|(_, par, _)| *par == k).collect();
                if cands.is_empty() {
                    continue;
                }
                let (cost, _, _, _, _, _) = stats(k);
                let ok = if combat {
                    stars >= cost as i64 && my_units_total < 12 + 3 * my_cities_list.len()
                } else if expanding {
                    stars >= cost as i64
                } else {
                    stars >= cost as i64 + 3
                };
                if !ok {
                    continue;
                }
                // prefer training city: closest to nearest unclaimed village (expansion) or enemy (army)
                let mut best: Option<(i32, String)> = None;
                for (t, _, id) in &cands {
                    let mut d = i32::MAX;
                    if want_army && !enemy_cities.is_empty() {
                        for &ec in &enemy_cities {
                            d = d.min(cheb(*t, ec, size));
                        }
                    } else {
                        for &vv in &villages {
                            d = d.min(cheb(*t, vv, size));
                        }
                    }
                    if d == i32::MAX {
                        d = 30;
                    }
                    // mild preference for bigger cities (more capacity later)
                    let lvl = my_city_level.get(t).copied().unwrap_or(1);
                    let d = d - lvl as i32;
                    if best.as_ref().map_or(true, |(bd, _)| d < *bd) {
                        best = Some((d, id.clone()));
                    }
                }
                if let Some((_, id)) = best {
                    return id;
                }
            }
        }

        // ============ 9. movement ============
        {
            // passability masks
            let climbing = tech[20];
            let mut enter_land = vec![false; n];
            let mut expand_land = vec![false; n];
            let mut enter_boat = vec![false; n];
            let mut expand_boat = vec![false; n];
            for i in 0..n {
                let t = &tiles[i];
                let land = t.terrain == 3 || t.terrain == 5 || (t.terrain == 4 && climbing);
                let unknown = !t.known;
                let port = t.structure == 8;
                let water = t.terrain == 1 || t.terrain == 2;
                enter_land[i] = unknown || land || port;
                expand_land[i] = land;
                enter_boat[i] = water || port || land || unknown;
                expand_boat[i] = water || unknown;
            }

            // fields (land)
            let mut field_village = bfs(n, size, &villages, &enter_land, &expand_land);
            let field_ruin = bfs(n, size, &ruins, &enter_land, &expand_land);
            let field_enemy = bfs(n, size, &enemy_cities, &enter_land, &expand_land);
            // frontier: known passable land tiles adjacent to unknown tiles
            let frontier: Vec<usize> = (0..n)
                .filter(|&i| {
                    tiles[i].known
                        && enter_land[i]
                        && cheb_neighbors(i, size, n).iter().any(|&j| !tiles[j].known)
                })
                .collect();
            let field_frontier = bfs(n, size, &frontier, &enter_land, &expand_land);
            let field_threat = bfs(n, size, &threat, &enter_land, &expand_land);
            let field_garrison = bfs(n, size, &garrisons, &enter_land, &expand_land);
            let own_ports: Vec<usize> = (0..n)
                .filter(|&i| tiles[i].known && tiles[i].structure == 8 && tiles[i].owner == me)
                .collect();
            let field_port = bfs(n, size, &own_ports, &enter_land, &expand_land);
            let field_home = bfs(n, size, &my_cities_list, &enter_land, &expand_land);
            let front: Vec<usize> = units
                .iter()
                .filter(|u| u.owner == me && enemy_cities.iter().any(|&ec| cheb(u.tile, ec, size) <= 6))
                .map(|u| u.tile)
                .collect();
            let field_front = bfs(n, size, &front, &enter_land, &expand_land);

            // boat fields: goal = land tiles near enemy cities / unknown land
            let boat_goals: Vec<usize> = if !enemy_cities.is_empty() {
                let mut g: Vec<usize> = enemy_cities.clone();
                for &ec in &enemy_cities {
                    for &j in &cheb_neighbors(ec, size, n) {
                        g.push(j);
                    }
                }
                g
            } else {
                Vec::new()
            };
            let field_boat_enemy = bfs(n, size, &boat_goals, &enter_boat, &expand_boat);
            let boat_frontier: Vec<usize> = (0..n)
                .filter(|&i| enter_boat[i] && cheb_neighbors(i, size, n).iter().any(|&j| !tiles[j].known))
                .collect();
            let field_boat_frontier = bfs(n, size, &boat_frontier, &enter_boat, &expand_boat);

            let big_inf = i32::MAX / 4;
            loop {
                // pick best (unit, goal-type)
                let mut best: Option<(f32, usize, u8, String)> = None; // score, unit tile, goal type, action id
                for u in units.iter().filter(|u| u.owner == me && !u.moved) {
                    let dests = match moves.get(&u.tile) {
                        Some(d) if !d.is_empty() => d,
                        _ => continue,
                    };
                    let _ = dests;
                    let is_boat = tiles[u.tile].terrain <= 2 || u.kind == 43;
                    // let wounded units on friendly land heal (end-of-turn +4)
                    let (_, umax, _, _, _, _) = stats(u.kind);
                    let umax = umax + if u.vet { 5.0 } else { 0.0 };
                    if !is_boat
                        && u.hp < 0.55 * umax
                        && tiles[u.tile].owner == me
                        && !threat.iter().any(|&t| cheb(t, u.tile, size) <= 2)
                    {
                        continue;
                    }
                    let mut bs: f32 = -1.0;
                    let mut btype: u8 = 0;
                    if !is_boat {
                        let dg = field_garrison[u.tile];
                        if dg < big_inf {
                            let s = 112.0 - 8.0 * dg as f32;
                            if s > bs {
                                bs = s;
                                btype = 6;
                            }
                        }
                        let dt = field_threat[u.tile];
                        if dt < big_inf {
                            let s = 98.0 - 8.0 * dt as f32;
                            if s > bs {
                                bs = s;
                                btype = 5;
                            }
                        }
                        let dv = field_village[u.tile];
                        if dv < big_inf {
                            let s = 100.0 - 7.0 * dv as f32;
                            if s > bs {
                                bs = s;
                                btype = 1;
                            }
                        }
                        let dr = field_ruin[u.tile];
                        if dr < big_inf {
                            let s = 55.0 - 7.0 * dr as f32;
                            if s > bs {
                                bs = s;
                                btype = 2;
                            }
                        }
                        let de = field_enemy[u.tile];
                        let lm = units
                            .iter()
                            .filter(|x| x.owner == me && cheb(x.tile, u.tile, size) <= 4)
                            .count();
                        let le = units
                            .iter()
                            .filter(|x| x.owner == enemy_seat && cheb(x.tile, u.tile, size) <= 4)
                            .count();
                        let may_assault = !outnumbered
                            && my_units_total >= 4
                            && (le == 0 || lm >= le + 1);
                        if de < big_inf && may_assault {
                            let s = 95.0 - 6.0 * de as f32 + if enemy_unit_seen { 10.0 } else { 0.0 };
                            if s > bs {
                                bs = s;
                                btype = 3;
                            }
                        }
                        let df = field_frontier[u.tile];
                        if df < big_inf && !outnumbered {
                            let s = 42.0 - 6.0 * df as f32;
                            if s > bs {
                                bs = s;
                                btype = 4;
                            }
                        }
                        if outnumbered {
                            let dh = field_home[u.tile];
                            if dh < big_inf {
                                let s = 48.0 - 5.0 * dh as f32;
                                if s > bs {
                                    bs = s;
                                    btype = 8;
                                }
                            }
                        }
                    } else {
                        let de = field_boat_enemy[u.tile];
                        if de < big_inf && !outnumbered {
                            let s = 95.0 - 6.0 * de as f32;
                            if s > bs {
                                bs = s;
                                btype = 3;
                            }
                        }
                        let df = field_boat_frontier[u.tile];
                        if df < big_inf {
                            let s = 45.0 - 6.0 * df as f32;
                            if s > bs {
                                bs = s;
                                btype = 4;
                            }
                        }
                    }
                    if my_city_level.contains_key(&u.tile)
                        && !threatened_cities.contains(&u.tile)
                    {
                        bs += 15.0; // vacate city tiles so we can train
                    }
                    if threatened_cities.contains(&u.tile) {
                        bs -= 60.0; // hold the walls
                    }
                    if bs <= 0.0 && !is_boat && enemy_city_known && !outnumbered {
                        let dh = field_front[u.tile];
                        if dh < big_inf {
                            let s = 46.0 - 5.0 * dh as f32;
                            if s > bs {
                                bs = s;
                                btype = 9;
                            }
                        }
                    }
                    if bs <= 0.0 && !is_boat && !own_ports.is_empty() {
                        // nothing to do on land: head to a port and sail
                        let dp = field_port[u.tile];
                        if dp < big_inf {
                            let urgent = enemy_city_known && field_enemy[u.tile] >= big_inf;
                            let s = if urgent { 75.0 - 6.0 * dp as f32 } else { 34.0 - 5.0 * dp as f32 };
                            if s > bs {
                                bs = s;
                                btype = 7;
                            }
                        }
                    }
                    if bs > 0.0 && best.as_ref().map_or(true, |(s, _, _, _)| bs > *s) {
                        // find the destination now
                        let field = match btype {
                            1 => &field_village,
                            2 => &field_ruin,
                            3 => if is_boat { &field_boat_enemy } else { &field_enemy },
                            5 => &field_threat,
                            6 => &field_garrison,
                            7 => &field_port,
                            8 => &field_home,
                            9 => &field_front,
                            _ => if is_boat { &field_boat_frontier } else { &field_frontier },
                        };
                        let dests = moves.get(&u.tile).cloned().unwrap_or_default();
                        let mut best_d: Option<(i32, i32, usize)> = None; // key: (dist, -tb, dst) minimized
                        for &d in dests.iter() {
                            if d >= n {
                                continue;
                            }
                            let fd = if field[d] < big_inf { field[d] } else { 60 };
                            let mut tb: i32 = 0;
                            if tiles[d].structure == 1 && tiles[d].owner == 0 {
                                tb += 100; // step onto village
                            }
                            if tiles[d].structure == 2 {
                                tb += 80;
                            }
                            if tiles[d].terrain == 4 || tiles[d].terrain == 5 {
                                tb += 3;
                            }
                            if tiles[d].owner == enemy_seat {
                                tb += 5;
                            }
                            // avoid parking next to multiple enemies
                            let adj_enemy = cheb_neighbors(d, size, n)
                                .iter()
                                .filter(|&&j| units.iter().any(|e| e.owner == enemy_seat && e.tile == j))
                                .count();
                            tb -= adj_enemy as i32 * 4;
                            let key = (fd, -tb, d as i32);
                            if best_d.map_or(true, |(a, b, _)| key.0 < a || (key.0 == a && key.1 < b)) {
                                best_d = Some((fd, -tb, d));
                            }
                        }
                        if let Some((_, _, dst)) = best_d {
                            if let Some(a) = move_ids.get(&(u.tile, dst as usize)) {
                                best = Some((bs, u.tile, btype, a.clone()));
                            }
                        }
                    }
                }
                match best {
                    Some((score, ut, btype, id)) => {
                        if score <= 0.0 {
                            break;
                        }
                        // claim the nearest village if that was the goal
                        if btype == 1 {
                            let mut bi: Option<(i32, usize)> = None;
                            for &vv in villages.iter() {
                                let d = cheb(ut, vv, size);
                                if bi.map_or(true, |(bd, _)| d < bd) {
                                    bi = Some((d, vv));
                                }
                            }
                            if let Some((_, vv)) = bi {
                                villages.retain(|&x| x != vv);
                                self.claimed.insert(vv);
                                field_village = bfs(n, size, &villages, &enter_land, &expand_land);
                            }
                        }
                        return id;
                    }
                    None => break,
                }
            }
        }

        // ============ 10. end turn ============
        if let Some(id) = end_id {
            return id;
        }
        first_id.unwrap_or_else(|| "0".to_string())
    }
}

fn cheb_neighbors(i: usize, size: usize, _n: usize) -> Vec<usize> {
    let (x, y) = ((i % size) as i32, (i / size) as i32);
    let mut out = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && nx < size as i32 && ny < size as i32 {
                out.push((ny as usize) * size + nx as usize);
            }
        }
    }
    out
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut bot = Bot::new();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let obs: Value = match serde_json::from_str(&line) {
            Ok(o) => o,
            Err(_) => continue,
        };
        let id = bot.choose(&obs);
        let _ = writeln!(stdout, "{}", json!({"action_id": id}));
        let _ = stdout.flush();
    }
}
