// Polytopia Bardur 1v1 bot: tiered greedy with BFS objective fields, memory, light navy.
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::io::{self, BufRead, Write};

const BIG: i32 = 1_000_000;

#[derive(Clone)]
struct Act {
    id: String,
    kind: i64,
    src: i64,
    dst: i64,
    param: i64,
}

fn cheb(a: usize, b: usize, size: usize) -> i32 {
    let dx = (a % size) as i32 - (b % size) as i32;
    let dy = (a / size) as i32 - (b / size) as i32;
    dx.abs().max(dy.abs())
}

// (attack, defense, maxhp, range, cost)
fn ustat(kind: i64) -> (f32, f32, f32, i32, i32) {
    match kind {
        2 => (2.0, 2.0, 10.0, 1, 2),
        3 => (2.0, 1.0, 10.0, 1, 3),
        4 => (3.5, 1.0, 10.0, 1, 8),
        5 => (1.0, 3.0, 15.0, 1, 3),
        8 => (4.0, 0.0, 10.0, 3, 8),
        9 => (2.0, 1.0, 10.0, 2, 3),
        10 => (0.0, 1.0, 10.0, 1, 5),
        11 => (3.0, 3.0, 15.0, 1, 5),
        12 => (5.0, 4.0, 40.0, 1, 10),
        15 => (3.0, 1.0, 15.0, 1, 2),
        42 => (2.0, 1.0, 10.0, 2, 5),
        43 => (0.0, 1.0, 10.0, 2, 0),
        44 => (3.0, 3.0, 10.0, 1, 5),
        45 => (4.0, 4.0, 40.0, 1, 10),
        41 => (3.0, 2.0, 10.0, 3, 15),
        _ => (2.0, 2.0, 10.0, 1, 4),
    }
}

fn is_stiff(kind: i64) -> bool {
    matches!(kind, 8 | 10 | 22 | 38 | 39 | 41 | 43)
}

fn is_float(kind: i64) -> bool {
    matches!(kind, 41 | 42 | 43 | 44 | 45 | 39 | 40)
}

fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    if total <= 0.0 {
        return (0.0, 0.0);
    }
    let atk = ((af / total) * aa * 4.5 + 0.000001 + 0.5).floor();
    let def = if dh > atk { ((df / total) * dd * 4.5 + 0.000001 + 0.5).floor() } else { 0.0 };
    (atk, def)
}

fn tech_tier(t: i64) -> i32 {
    match t {
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 23 | 44 => 2,
        1 | 6 | 10 | 15 | 20 | 25 | 48 => 1,
        _ => 3,
    }
}

fn bfs(sources: &[usize], pass: &[bool], size: usize) -> Vec<i32> {
    let n = pass.len();
    let mut dist = vec![BIG; n];
    let mut q: VecDeque<usize> = VecDeque::new();
    for &s in sources {
        if s < n && dist[s] == BIG {
            dist[s] = 0;
            q.push_back(s);
        }
    }
    while let Some(i) = q.pop_front() {
        let x = (i % size) as i32;
        let y = (i / size) as i32;
        let d = dist[i] + 1;
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = x + dx;
                let ny = y + dy;
                if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 {
                    continue;
                }
                let j = (ny as usize) * size + nx as usize;
                if dist[j] == BIG && pass[j] {
                    dist[j] = d;
                    q.push_back(j);
                }
            }
        }
    }
    dist
}

#[derive(Clone)]
struct MyU {
    kind: i64,
    hp: f32,
    vet: bool,
    passenger: i64,
}
#[derive(Clone)]
struct EnU {
    kind: i64,
    hp: f32,
    vet: bool,
}

struct Bot {
    size: usize,
    n: usize,
    terrain: Vec<i32>,
    structure: Vec<i32>,
    owner: Vec<i32>,
    resource: Vec<i32>,
    seen: Vec<bool>,
    enemy_capital: Option<usize>,
}

impl Bot {
    fn new() -> Bot {
        Bot {
            size: 0,
            n: 0,
            terrain: Vec::new(),
            structure: Vec::new(),
            owner: Vec::new(),
            resource: Vec::new(),
            seen: Vec::new(),
            enemy_capital: None,
        }
    }

    fn update(&mut self, v: &Value, enemy: i32) {
        let size = v["board_size"].as_u64().unwrap_or(14) as usize;
        if size == 0 {
            return;
        }
        if self.n == 0 {
            self.size = size;
            self.n = size * size;
            self.terrain = vec![-1; self.n];
            self.structure = vec![0; self.n];
            self.owner = vec![0; self.n];
            self.resource = vec![0; self.n];
            self.seen = vec![false; self.n];
        }
        if let Some(arr) = v["tiles"].as_array() {
            for t in arr {
                let i = t["tile"].as_u64().unwrap_or(0) as usize;
                if i >= self.n || t["known"].as_bool() != Some(true) {
                    continue;
                }
                self.seen[i] = true;
                if let Some(x) = t["terrain"].as_i64() {
                    self.terrain[i] = x as i32;
                }
                if let Some(x) = t["structure"].as_i64() {
                    self.structure[i] = x as i32;
                }
                if let Some(x) = t["owner"].as_i64() {
                    self.owner[i] = x as i32;
                }
                if let Some(x) = t["resource"].as_i64() {
                    self.resource[i] = x as i32;
                }
                if t["capital"].as_i64() == Some(enemy as i64) {
                    self.enemy_capital = Some(i);
                }
            }
        }
    }

    fn choose(&mut self, obs: &Value) -> String {
        let v = &obs["visible_state"];
        let player = obs["player"].as_i64().unwrap_or(1) as i32;
        let enemy = 3 - player;
        self.update(v, enemy);
        let n = self.n;
        let size = self.size;
        if n == 0 {
            return obs["legal_actions"][0]["action_id"].as_str().unwrap_or("0").to_string();
        }
        let turn = v["turn"].as_i64().unwrap_or(0) as i32;
        let stars = v["stars"].as_i64().unwrap_or(0) as i32;
        let mut tech = [false; 128];
        if let Some(arr) = v["technology"].as_array() {
            for t in arr {
                if let Some(k) = t.as_i64() {
                    if k >= 0 && k < 128 {
                        tech[k as usize] = true;
                    }
                }
            }
        }
        let mut legal: Vec<Act> = Vec::new();
        if let Some(arr) = obs["legal_actions"].as_array() {
            for a in arr {
                let c = &a["command"];
                legal.push(Act {
                    id: a["action_id"].as_str().unwrap_or("0").to_string(),
                    kind: c["kind"].as_i64().unwrap_or(0),
                    src: c["source"].as_i64().unwrap_or(-1),
                    dst: c["target"].as_i64().unwrap_or(-1),
                    param: c["parameter"].as_i64().unwrap_or(0),
                });
            }
        }
        if legal.is_empty() {
            return "0".to_string();
        }
        let fallback = || -> String {
            for a in &legal {
                if a.kind == 10 {
                    return a.id.clone();
                }
            }
            for a in &legal {
                if a.kind != 11 {
                    return a.id.clone();
                }
            }
            legal[0].id.clone()
        };

        // ---- units ----
        let mut myu: Vec<Option<MyU>> = vec![None; n];
        let mut enu: Vec<Option<EnU>> = vec![None; n];
        let mut my_count = 0i32;
        let mut my_val = 0i32;
        let mut en_val = 0i32;
        if let Some(arr) = v["units"].as_array() {
            for u in arr {
                let tile = u["tile"].as_u64().unwrap_or(0) as usize;
                if tile >= n {
                    continue;
                }
                let kind = u["kind"].as_i64().unwrap_or(2);
                let hp = u["health"].as_f64().unwrap_or(10.0) as f32;
                let vet = u["veteran"].as_bool().unwrap_or(false);
                let owner = u["owner"].as_i64().unwrap_or(0) as i32;
                let cost = ustat(kind).4;
                if owner == player {
                    my_count += 1;
                    my_val += cost;
                    myu[tile] = Some(MyU { kind, hp, vet, passenger: u["passenger"].as_i64().unwrap_or(0) });
                } else if owner == enemy {
                    en_val += cost;
                    enu[tile] = Some(EnU { kind, hp, vet });
                }
            }
        }
        // ---- cities ----
        let mut my_cities: Vec<(usize, i32)> = Vec::new();
        if let Some(arr) = v["cities"].as_array() {
            for c in arr {
                let tile = c["tile"].as_u64().unwrap_or(0) as usize;
                if tile >= n {
                    continue;
                }
                if c["owner"].as_i64().unwrap_or(0) == player as i64 {
                    my_cities.push((tile, c["level"].as_i64().unwrap_or(1) as i32));
                }
            }
        }
        let my_cities_n = my_cities.len() as i32;
        let mut en_city_tiles: Vec<usize> = Vec::new();
        let mut villages: Vec<usize> = Vec::new();
        let mut ruins: Vec<usize> = Vec::new();
        let mut ports: Vec<usize> = Vec::new();
        for i in 0..n {
            if !self.seen[i] {
                continue;
            }
            let st = self.structure[i];
            if st == 1 {
                if self.owner[i] == enemy {
                    en_city_tiles.push(i);
                } else if self.owner[i] == 0 {
                    villages.push(i);
                }
            } else if st == 2 {
                ruins.push(i);
            } else if st == 8 && self.owner[i] == player {
                ports.push(i);
            }
        }
        let en_cities_n = en_city_tiles.len() as i32;

        // local balance near our cities
        let mut near_en = 0i32;
        let mut near_my = 0i32;
        for (j, e) in enu.iter().enumerate() {
            if let Some(e) = e {
                if my_cities.iter().any(|&(c, _)| cheb(c, j, size) <= 4) {
                    near_en += ustat(e.kind).4;
                }
            }
        }
        for (j, m) in myu.iter().enumerate() {
            if let Some(m) = m {
                if my_cities.iter().any(|&(c, _)| cheb(c, j, size) <= 4) {
                    near_my += ustat(m.kind).4;
                }
            }
        }
        let defensive = near_en > near_my * 2 + 6;
        // attack mode?
        let mut attack_mode = !defensive && (my_val >= en_val + 3 || (turn >= 20 && my_cities_n > en_cities_n));
        if myu.iter().any(|u| u.as_ref().map_or(false, |u| u.kind == 12)) {
            attack_mode = true;
        }

        // relations / meeting
        let mut met = false;
        if let Some(arr) = v["relations"]["known_players"].as_array() {
            let idx = (enemy - 1) as usize;
            if idx < arr.len() {
                met = arr[idx].as_bool().unwrap_or(false);
            }
        }

        // passability (land, incl. own ports and bridges)
        let climb = tech[20];
        let mut pass = vec![false; n];
        for i in 0..n {
            let t = self.terrain[i];
            pass[i] = match t {
                -1 => true,
                3 | 5 => true,
                4 => climb,
                1 | 2 => (self.structure[i] == 8 && self.owner[i] == player) || self.structure[i] == 48,
                _ => true,
            };
        }
        // water-inclusive passability (for reachability checks)
        let mut pass_all = vec![false; n];
        for i in 0..n {
            let t = self.terrain[i];
            pass_all[i] = match t {
                -1 => true,
                3 | 5 => true,
                4 => climb,
                _ => true,
            };
        }

        // naval?
        let mut naval = false;
        let known_count = self.seen.iter().filter(|&&s| s).count();
        let water_known = (0..n).filter(|&i| self.seen[i] && self.terrain[i] <= 2 && self.terrain[i] > 0).count();
        if !met && turn >= 8 && known_count * 3 > n && known_count > 0 && water_known * 5 > known_count {
            naval = true;
        }
        if let Some(ec) = self.enemy_capital {
            if let Some(&(cap, _)) = my_cities.first() {
                let fl = bfs(&[cap], &pass, size);
                let fw = bfs(&[cap], &pass_all, size);
                if fl[ec] >= BIG && fw[ec] < BIG {
                    naval = true;
                }
            }
        }

        // ---------------- tier 1: promote (free veteran) ----------------
        for a in &legal {
            if a.kind == 3 && a.param == 16 {
                return a.id.clone();
            }
        }

        // ---------------- tier 2: city rewards ----------------
        let mut best_reward: Option<(i32, &Act)> = None;
        for a in &legal {
            if a.kind != 9 {
                continue;
            }
            let pref = match a.param {
                6 => 100,
                3 => 90,
                7 => 80,
                8 => 70,
                5 => 60,
                1 => 40,
                2 => 55,
                _ => 30,
            };
            let mut sc = pref;
            let t = a.dst as usize;
            if a.param == 1 && t < n {
                // wall is great when the city is threatened
                let threatened = enu.iter().enumerate().any(|(j, e)| e.is_some() && cheb(j, t, size) <= 2);
                if threatened {
                    sc += 100;
                }
            }
            if best_reward.as_ref().map_or(true, |(bs, _)| sc > *bs) {
                best_reward = Some((sc, a));
            }
        }
        if let Some((_, a)) = best_reward {
            return a.id.clone();
        }

        // ---------------- tier 3: capture ----------------
        let mut best_cap: Option<(i32, &Act)> = None;
        for a in &legal {
            if a.kind != 8 {
                continue;
            }
            let s = a.src as usize;
            if s >= n {
                continue;
            }
            let sc = if self.structure[s] == 1 && self.owner[s] == enemy {
                100
            } else if self.structure[s] == 2 {
                60
            } else if self.structure[s] == 1 {
                80
            } else {
                40
            };
            if best_cap.as_ref().map_or(true, |(bs, _)| sc > *bs) {
                best_cap = Some((sc, a));
            }
        }
        if let Some((_, a)) = best_cap {
            return a.id.clone();
        }

        // ---------------- tier 4: attacks ----------------
        let mut best_atk: Option<(f32, &Act)> = None;
        for a in &legal {
            if a.kind != 2 {
                continue;
            }
            let (s, d) = (a.src as usize, a.dst as usize);
            if s >= n || d >= n {
                continue;
            }
            let (Some(m), Some(e)) = (myu[s].as_ref(), enu[d].as_ref()) else { continue };
            let (aa, _, am, _, acost) = ustat(m.kind);
            let am = am + if m.vet { 5.0 } else { 0.0 };
            let (_, dd, dm, drange, dcost) = ustat(e.kind);
            let dm = dm + if e.vet { 5.0 } else { 0.0 };
            // defense bonus of the defender's tile
            let mut db = 1.0;
            let terr = self.terrain[d];
            if terr == 4 || terr == 5 || (terr == 1 || terr == 2) {
                db = 1.5;
            } else if self.structure[d] == 1 && self.owner[d] == enemy {
                db = 1.5;
            }
            let (dmg, ret) = combat(aa, m.hp, am, dd, e.hp, dm, db);
            let kill = e.hp - dmg <= 0.0;
            let can_ret = !kill && !is_stiff(e.kind) && cheb(s, d, size) <= drange;
            let retv = if can_ret { ret } else { 0.0 };
            let mut sc = dmg * 3.0;
            if kill {
                sc += 25.0 + dcost as f32 * 5.0;
            }
            sc -= retv * 2.0;
            if m.hp - retv <= 0.0 {
                sc -= acost as f32 * 6.0;
            }
            if self.structure[d] == 1 {
                sc += 8.0;
            }
            let on_our_city = self.structure[d] == 1 && self.owner[d] == player;
            if on_our_city {
                sc += 40.0; // break the siege on our own city at (almost) any cost
            }
            if my_cities.iter().any(|&(c, _)| cheb(c, d, size) <= 2) {
                sc += 10.0;
            }
            if attack_mode {
                sc += 3.0;
            }
            if best_atk.as_ref().map_or(true, |(bs, _)| sc > *bs) {
                best_atk = Some((sc, a));
            }
        }
        if let Some((sc, a)) = best_atk {
            let d = a.dst as usize;
            let on_our_city = d < n && self.structure[d] == 1 && self.owner[d] == player;
            let thresh = if on_our_city { 0.0 } else if defensive { 14.0 } else { 6.0 };
            if sc >= thresh {
                return a.id.clone();
            }
        }

        // ---------------- tier 5: harvest ----------------
        for a in &legal {
            if a.kind == 5 {
                return a.id.clone();
            }
        }

        // ---------------- tier 6: train units ----------------
        let target_units = if attack_mode {
            (6 + 3 * my_cities_n).min(26)
        } else {
            (3 + 2 * my_cities_n).min(20)
        };
        if my_count < target_units || stars >= 10 {
            let mut best_train: Option<(i32, &Act)> = None;
            for a in &legal {
                if a.kind != 4 {
                    continue;
                }
                let (s, p) = (a.src as usize, a.param);
                if s >= n {
                    continue;
                }
                let mut sc = 0;
                if p == 11 && stars >= 8 {
                    sc = 50;
                } else if p == 3 && stars >= 5 {
                    sc = 45;
                } else if p == 2 && stars >= 2 {
                    sc = 40;
                } else if p == 9 && stars >= 5 {
                    sc = 35;
                } else if p == 5 && stars >= 5 {
                    sc = 30;
                } else if p == 4 && stars >= 10 {
                    sc = 28;
                } else if p == 15 && stars >= 4 {
                    sc = 25;
                } else {
                    continue;
                }
                // prefer bigger cities
                if let Some(&(_, lv)) = my_cities.iter().find(|&&(c, _)| c == s) {
                    sc += lv;
                }
                if best_train.as_ref().map_or(true, |(bs, _)| sc > *bs) {
                    best_train = Some((sc, a));
                }
            }
            if let Some((_, a)) = best_train {
                return a.id.clone();
            }
        }

        // ---------------- tier 7: research ----------------
        {
            let mut prio: Vec<i64> = Vec::new();
            if naval && !tech[10] {
                prio.push(10);
            }
            for t in [6i64, 20, 1, 10, 16, 23, 8, 18, 24, 4, 7, 13, 17, 3, 9, 22, 5, 2, 19, 21, 12, 14, 44, 38] {
                if !prio.contains(&t) {
                    prio.push(t);
                }
            }
            let reserve = if my_count <= 2 { 4 } else { 2 };
            for t in prio {
                if tech[t as usize] {
                    continue;
                }
                let mut cost = 4 + my_cities_n * tech_tier(t);
                if tech[22] {
                    cost = ((cost as f32) * 0.66667).ceil() as i32;
                }
                if stars >= cost + reserve {
                    for a in &legal {
                        if a.kind == 7 && a.param == t {
                            return a.id.clone();
                        }
                    }
                }
            }
        }

        // ---------------- tier 8: build ----------------
        if turn >= 5 {
            let mut best_build: Option<(i32, &Act)> = None;
            for a in &legal {
                if a.kind != 6 {
                    continue;
                }
                let (t, p) = (a.dst as usize, a.param);
                if t >= n {
                    continue;
                }
                let (cost, mut sc) = match p {
                    8 => (7, 15),
                    5 => (5, 24),
                    21 => (5, 24),
                    12 => (3, 20),
                    6 => (5, 18),
                    13 => (5, 18),
                    22 => (5, 18),
                    50 => (5, 16),
                    3 => (3, 6),
                    48 => (5, 10),
                    17 => (20, 4),
                    18 => (15, 4),
                    19 | 20 => (20, 4),
                    _ => (5, 8),
                };
                if p == 12 && self.resource[t] == 1 {
                    continue; // don't pave over animals
                }
                if p == 8 && self.resource[t] == 8 {
                    continue; // don't pave over starfish
                }
                if naval && p == 8 && (ports.len() as i32) < 2 {
                    sc = 200;
                    // prefer ports facing the enemy / frontier
                    if let Some(ec) = self.enemy_capital {
                        sc -= cheb(t, ec, size).min(30);
                    }
                    if stars >= cost {
                        if best_build.as_ref().map_or(true, |(bs, _)| sc > *bs) {
                            best_build = Some((sc, a));
                        }
                    }
                    continue;
                }
                if stars >= cost + 6 {
                    if best_build.as_ref().map_or(true, |(bs, _)| sc > *bs) {
                        best_build = Some((sc, a));
                    }
                }
            }
            if let Some((_, a)) = best_build {
                return a.id.clone();
            }
        }

        // ---------------- tier 9: moves ----------------
        {
            // frontier: unknown tiles adjacent to known ones
            let mut frontier: Vec<usize> = Vec::new();
            for i in 0..n {
                if self.seen[i] {
                    continue;
                }
                let x = (i % size) as i32;
                let y = (i / size) as i32;
                let mut adj_known = false;
                'outer: for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = x + dx;
                        let ny = y + dy;
                        if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 {
                            continue;
                        }
                        if self.seen[(ny as usize) * size + nx as usize] {
                            adj_known = true;
                            break 'outer;
                        }
                    }
                }
                if adj_known {
                    frontier.push(i);
                }
            }
            let mut expand_src: Vec<usize> = Vec::new();
            for &t in villages.iter().chain(ruins.iter()) {
                if t < n && pass[t] {
                    expand_src.push(t);
                }
            }
            let f_village = bfs(&expand_src, &pass, size);
            let f_explore = bfs(&frontier, &pass, size);
            let f_enemy = if en_city_tiles.is_empty() {
                vec![BIG; n]
            } else {
                bfs(&en_city_tiles, &pass, size)
            };
            // threats: enemy units near our cities
            let mut threats: Vec<usize> = Vec::new();
            for (j, e) in enu.iter().enumerate() {
                if e.is_none() {
                    continue;
                }
                if my_cities.iter().any(|&(c, _)| cheb(c, j, size) <= 3) {
                    threats.push(j);
                }
            }
            let f_threat = if threats.is_empty() {
                vec![BIG; n]
            } else if defensive {
                let home: Vec<usize> = my_cities.iter().map(|&(c, _)| c).collect();
                bfs(&home, &pass, size)
            } else {
                bfs(&threats, &pass, size)
            };
            let port_src: Vec<usize> = ports.iter().copied().filter(|&t| pass[t]).collect();
            let f_port = if naval && !port_src.is_empty() { bfs(&port_src, &pass, size) } else { vec![BIG; n] };

            let mut objv = vec![BIG; n];
            for i in 0..n {
                let mut m = BIG;
                if f_village[i] < BIG {
                    m = m.min(f_village[i]);
                }
                if f_explore[i] < BIG {
                    m = m.min(f_explore[i] + 4);
                }
                if f_enemy[i] < BIG {
                    m = m.min(f_enemy[i] + if attack_mode { 0 } else { 12 });
                }
                if f_threat[i] < 6 {
                    m = m.min(f_threat[i] - 3);
                }
                if naval && f_port[i] < BIG {
                    m = m.min(f_port[i] + 6);
                }
                objv[i] = m;
            }

            // raft goal set (chebyshev): enemy cities, else frontier
            let raft_goals: Vec<usize> = if !en_city_tiles.is_empty() {
                en_city_tiles.clone()
            } else {
                frontier.clone()
            };

            let mut best_mv: Option<(i32, &Act)> = None;
            for a in &legal {
                if a.kind != 1 {
                    continue;
                }
                let (s, d) = (a.src as usize, a.dst as usize);
                if s >= n || d >= n {
                    continue;
                }
                let Some(m) = myu[s].as_ref() else { continue };
                let mut sc = 0i32;
                if is_float(m.kind) {
                    if raft_goals.is_empty() {
                        continue;
                    }
                    let gs = raft_goals.iter().map(|&g| cheb(s, g, size)).min().unwrap_or(BIG);
                    let gd = raft_goals.iter().map(|&g| cheb(d, g, size)).min().unwrap_or(BIG);
                    sc = 12 * (gs - gd);
                    if self.structure[d] == 1 && self.owner[d] != player {
                        sc += 25;
                    }
                    if self.structure[d] == 2 {
                        sc += 20;
                    }
                } else {
                    let os = if objv[s] >= BIG { 999 } else { objv[s] };
                    let od = if objv[d] >= BIG { 999 } else { objv[d] };
                    sc = 12 * (os - od);
                    if self.structure[d] == 1 && self.owner[d] == 0 {
                        sc += 25;
                    }
                    if self.structure[d] == 2 {
                        sc += 20;
                    }
                    if self.structure[d] == 1 && self.owner[d] == enemy {
                        sc += 25;
                    }
                    if self.structure[d] == 8 && !naval {
                        sc -= 60; // don't accidentally embark
                    }
                    // reveal bonus
                    let x = (d % size) as i32;
                    let y = (d / size) as i32;
                    let mut unk = 0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = x + dx;
                            let ny = y + dy;
                            if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 {
                                continue;
                            }
                            if !self.seen[(ny as usize) * size + nx as usize] {
                                unk += 1;
                            }
                        }
                    }
                    sc += unk * 2;
                    // hurt units drift home
                    if m.hp < 6.0 && my_cities.iter().any(|&(c, _)| c == d) {
                        sc += 8;
                    }
                    // garrison bonus when defensive
                    if defensive && my_cities.iter().any(|&(c, _)| c == d) {
                        sc += 6;
                    }
                }
                if best_mv.as_ref().map_or(true, |(bs, _)| sc > *bs) {
                    best_mv = Some((sc, a));
                }
            }
            if let Some((sc, a)) = best_mv {
                if std::env::var("POLY_DEBUG").is_ok() {
                    eprintln!("best_move_score={} act_kind={} src={} dst={} objv_src={:?} fv={:?} fe={:?} fen={:?} npass={}", sc, a.kind, a.src, a.dst,
                        objv.get(a.src as usize).map(|&x| if x>=BIG {-1} else {x}),
                        f_village.get(a.src as usize).map(|&x| if x>=BIG {-1} else {x}),
                        f_explore.get(a.src as usize).map(|&x| if x>=BIG {-1} else {x}),
                        f_enemy.get(a.src as usize).map(|&x| if x>=BIG {-1} else {x}),
                        pass.iter().filter(|&&p| p).count());
                }
                if sc >= 3 {
                    return a.id.clone();
                }
            }
        }

        fallback()
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut bot = Bot::new();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obs: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let id = bot.choose(&obs);
        let _ = writeln!(out, "{}", json!({ "action_id": id }));
        let _ = out.flush();
    }
}
