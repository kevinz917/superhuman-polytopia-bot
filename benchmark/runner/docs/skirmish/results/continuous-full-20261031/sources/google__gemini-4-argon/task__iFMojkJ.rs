use serde_json::{json, Value};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering;
use std::io::{self, BufRead, Write};

#[derive(Clone, Debug)]
struct Tile {
    idx: usize,
    known: bool,
    terrain: i32, // 1 water, 2 ocean, 3 field, 4 mountain, 5 forest
    owner: i32,
    structure: i32, // 1 city/village, 2 ruin, 3 road, 5 farm, 6 windmill, 8 port, 12 hut, 13 sawmill, 21 mine, 22 forge, 47 lighthouse, 48 bridge, 50 market
    resource: i32,  // 1 animal, 2 crop, 3 fish, 5 metal, 6 fruit, 8 starfish
    road: bool,
    capital: i32,
    structure_level: i32,
}

#[derive(Clone, Debug)]
struct Unit {
    tile: usize,
    owner: i32,
    kind: i32,
    health: f32,
    veteran: bool,
    moved: bool,
    attacked: bool,
    kills: i32,
    home: i32,
    passenger: i32,
    poison: bool,
    frozen: bool,
    boosted: bool,
    invisible: bool,
    detect: bool,
}

#[derive(Clone, Debug)]
struct City {
    tile: usize,
    owner: i32,
    level: i32,
    population: i32,
    production: i32,
    progress: i32,
    connected: bool,
    riot: bool,
    rewards: [i32; 9],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Cmd {
    kind: i32,
    source: i32,
    target: i32,
    parameter: i32,
}

#[derive(Clone, Debug)]
struct Legal {
    id: String,
    cmd: Cmd,
}

#[derive(Clone, Debug)]
struct Obs {
    player: i32,
    enemy: i32,
    state_version: i32,
    size: usize,
    n: usize,
    turn: i32,
    stars: i32,
    tech: [bool; 128],
    tiles: Vec<Tile>,
    units: Vec<Unit>,
    unit_at: Vec<Option<usize>>, // index into units
    cities: Vec<City>,
    city_at: Vec<Option<usize>>, // index into cities
    peace: bool,
    peace_offered_to_me: bool,
    embassy_sent: bool,
    embassy_recv: bool,
    known_enemy: bool,
    legal: Vec<Legal>,
}

struct BotMemory {
    last_turn: i32,
    last_version: i32,
    enemy_capital: Option<usize>,
    my_capital: Option<usize>,
    known_walled_cities: HashSet<usize>,
    last_seen_enemy_units: HashMap<usize, (i32, f32, i32)>, // tile -> (kind, hp, turn)
    remembered_structures: HashMap<usize, i32>,
    remembered_owners: HashMap<usize, i32>,
    visited_targets: HashMap<usize, i32>,
}

impl BotMemory {
    fn new() -> Self {
        BotMemory {
            last_turn: -1,
            last_version: -1,
            enemy_capital: None,
            my_capital: None,
            known_walled_cities: HashSet::new(),
            last_seen_enemy_units: HashMap::new(),
            remembered_structures: HashMap::new(),
            remembered_owners: HashMap::new(),
            visited_targets: HashMap::new(),
        }
    }

    fn reset_if_new_game(&mut self, turn: i32, version: i32) {
        if (turn == 0 && version <= 2 && self.last_turn > 0) || version < self.last_version {
            *self = BotMemory::new();
        }
        self.last_turn = turn;
        self.last_version = version;
    }

    fn update(&mut self, obs: &Obs, raw: &Value) {
        self.reset_if_new_game(obs.turn, obs.state_version);
        for t in &obs.tiles {
            if t.known {
                if t.capital == obs.player || (self.my_capital.is_none() && obs.turn == 0 && t.owner == obs.player && t.structure == 1) {
                    self.my_capital = Some(t.idx);
                }
                if t.capital == obs.enemy {
                    self.enemy_capital = Some(t.idx);
                }
                self.remembered_structures.insert(t.idx, t.structure);
                self.remembered_owners.insert(t.idx, t.owner);
            }
        }
        if let Some(hist) = raw["remembered_state"]["historical"].as_array() {
            for h in hist {
                let entity = h["entity"].as_str().unwrap_or("");
                let fact = &h["fact"];
                if let Some(tile) = fact["tile"].as_i64() {
                    let ti = tile as usize;
                    if entity == "tiles" {
                        if let Some(st) = fact["structure"].as_i64() {
                            self.remembered_structures.entry(ti).or_insert(st as i32);
                        }
                        if let Some(ow) = fact["owner"].as_i64() {
                            self.remembered_owners.entry(ti).or_insert(ow as i32);
                        }
                        if fact["capital"].as_i64() == Some(obs.enemy as i64) {
                            self.enemy_capital = Some(ti);
                        }
                    } else if entity == "cities" {
                        if let Some(ow) = fact["owner"].as_i64() {
                            self.remembered_owners.entry(ti).or_insert(ow as i32);
                            self.remembered_structures.entry(ti).or_insert(1);
                        }
                    }
                }
            }
        }
        for u in &obs.units {
            if u.owner == obs.enemy {
                self.last_seen_enemy_units.insert(u.tile, (u.kind, u.health, obs.turn));
            } else {
                self.last_seen_enemy_units.remove(&u.tile);
            }
        }
    }
}

#[derive(Clone, Copy)]
struct UnitStats {
    cost: i32,
    health: f32,
    attack: f32,
    defense: f32,
    movement: i32,
    range: i32,
    dash: bool,
    escape: bool,
    persist: bool,
    fortify: bool,
    stiff: bool,
    stat: bool,
    float: bool,
    carry: bool,
    scout: bool,
    creep: bool,
    hide: bool,
    independent: bool,
    surprise: bool,
    splash: bool,
    stomp: bool,
    infiltrate: bool,
}

fn unit_stats(kind: i32) -> UnitStats {
    let mut s = UnitStats {
        cost: 0, health: 10.0, attack: 0.0, defense: 0.0, movement: 1, range: 1,
        dash: false, escape: false, persist: false, fortify: false, stiff: false, stat: false,
        float: false, carry: false, scout: false, creep: false, hide: false, independent: false,
        surprise: false, splash: false, stomp: false, infiltrate: false,
    };
    match kind {
        2 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.fortify = true; }
        3 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 2; s.range = 1; s.dash = true; s.escape = true; s.fortify = true; }
        4 => { s.cost = 8; s.health = 10.0; s.attack = 3.5; s.defense = 1.0; s.movement = 3; s.range = 1; s.dash = true; s.fortify = true; s.persist = true; }
        5 => { s.cost = 3; s.health = 15.0; s.attack = 1.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.fortify = true; }
        8 => { s.cost = 8; s.health = 10.0; s.attack = 4.0; s.defense = 0.0; s.movement = 1; s.range = 3; s.stiff = true; }
        9 => { s.cost = 3; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 1; s.range = 2; s.dash = true; s.fortify = true; }
        10 => { s.cost = 5; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 1; s.range = 1; s.stiff = true; }
        11 => { s.cost = 5; s.health = 15.0; s.attack = 3.0; s.defense = 3.0; s.movement = 1; s.range = 1; s.dash = true; }
        12 => { s.cost = 10; s.health = 40.0; s.attack = 5.0; s.defense = 4.0; s.movement = 1; s.range = 1; s.stat = true; }
        37 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 1; s.range = 1; s.dash = true; s.independent = true; s.stat = true; s.surprise = true; }
        38 => { s.cost = 8; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.creep = true; s.dash = true; s.hide = true; s.infiltrate = true; s.scout = true; s.stat = true; s.stiff = true; }
        39 => { s.cost = 2; s.health = 5.0; s.attack = 0.0; s.defense = 0.5; s.movement = 2; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.hide = true; s.infiltrate = true; s.scout = true; s.stat = true; s.stiff = true; }
        40 => { s.cost = 2; s.health = 10.0; s.attack = 2.0; s.defense = 2.0; s.movement = 2; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.independent = true; s.stat = true; s.surprise = true; }
        41 => { s.cost = 15; s.health = 10.0; s.attack = 3.0; s.defense = 2.0; s.movement = 2; s.range = 3; s.carry = true; s.float = true; s.splash = true; s.stat = true; s.stiff = true; }
        42 => { s.cost = 5; s.health = 10.0; s.attack = 2.0; s.defense = 1.0; s.movement = 3; s.range = 2; s.carry = true; s.dash = true; s.float = true; s.scout = true; s.stat = true; }
        43 => { s.cost = 0; s.health = 10.0; s.attack = 0.0; s.defense = 1.0; s.movement = 2; s.range = 2; s.carry = true; s.float = true; s.stat = true; s.stiff = true; }
        44 => { s.cost = 5; s.health = 10.0; s.attack = 3.0; s.defense = 3.0; s.movement = 3; s.range = 1; s.carry = true; s.dash = true; s.float = true; s.stat = true; }
        45 => { s.cost = 10; s.health = 40.0; s.attack = 4.0; s.defense = 4.0; s.movement = 2; s.range = 1; s.carry = true; s.float = true; s.stat = true; s.stiff = true; s.stomp = true; }
        _ => {}
    }
    s
}

fn tech_tier(tech_id: usize) -> i32 {
    match tech_id {
        0 => 0,
        1 | 6 | 10 | 15 | 20 => 1,
        2 | 4 | 7 | 8 | 13 | 16 | 18 | 21 | 23 | 44 => 2,
        3 | 5 | 9 | 12 | 14 | 17 | 19 | 22 | 24 | 38 => 3,
        _ => 1,
    }
}

fn research_cost_of(obs: &Obs, tech_id: usize, num_my_cities: i32) -> i32 {
    let base = 4 + num_my_cities * tech_tier(tech_id);
    if obs.tech[22] {
        ((base as f32) * 0.66666f32).ceil() as i32
    } else {
        base
    }
}

fn unit_max_hp(u: &Unit) -> f32 {
    let base_kind = if u.passenger > 0 { u.passenger } else { u.kind };
    let mut mh = unit_stats(base_kind).health + if u.veteran { 5.0 } else { 0.0 };
    if u.health > mh {
        // Enemy boat carrying a 15-HP unit (Defender/Swordsman) or Giant
        if u.veteran {
            mh = if u.health > 20.0 { 45.0 } else { 20.0 };
        } else {
            mh = if u.health > 15.0 { 40.0 } else { 15.0 };
        }
    }
    mh
}

fn unit_star_value(u: &Unit) -> f32 {
    let st = unit_stats(u.kind);
    let pass_cost = if u.passenger > 0 { unit_stats(u.passenger).cost } else { 0 };
    let base = (st.cost + pass_cost) as f32;
    if u.kind == 12 || u.kind == 45 {
        18.0
    } else {
        base.max(2.0)
    }
}

#[inline]
fn cheb(size: usize, a: usize, b: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs().max((ay - by).abs())
}

#[inline]
fn manhattan(size: usize, a: usize, b: usize) -> i32 {
    let (ax, ay) = ((a % size) as i32, (a / size) as i32);
    let (bx, by) = ((b % size) as i32, (b / size) as i32);
    (ax - bx).abs() + (ay - by).abs()
}

fn neighbors(size: usize, t: usize) -> Vec<usize> {
    let (x, y) = ((t % size) as i32, (t / size) as i32);
    let s = size as i32;
    let mut out = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && nx < s && ny >= 0 && ny < s {
                out.push((ny * s + nx) as usize);
            }
        }
    }
    out
}

fn ortho_neighbors(size: usize, t: usize) -> Vec<usize> {
    let (x, y) = ((t % size) as i32, (t / size) as i32);
    let s = size as i32;
    let mut out = Vec::with_capacity(4);
    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
        let (nx, ny) = (x + dx, y + dy);
        if nx >= 0 && nx < s && ny >= 0 && ny < s {
            out.push((ny * s + nx) as usize);
        }
    }
    out
}

fn parse_obs(v: &Value) -> Obs {
    let player = v["player"].as_i64().unwrap_or(1) as i32;
    let enemy = 3 - player;
    let state_version = v["state_version"].as_i64().unwrap_or(0) as i32;
    let vs = &v["visible_state"];
    let size = vs["board_size"].as_u64().unwrap_or(14) as usize;
    let n = size * size;
    let turn = vs["turn"].as_i64().unwrap_or(0) as i32;
    let stars = vs["stars"].as_i64().unwrap_or(0) as i32;
    let mut tech = [false; 128];
    if let Some(arr) = vs["technology"].as_array() {
        for x in arr {
            if let Some(id) = x.as_u64() {
                if (id as usize) < 128 {
                    tech[id as usize] = true;
                }
            }
        }
    }
    let mut tiles = vec![
        Tile {
            idx: 0, known: false, terrain: 0, owner: 0, structure: 0,
            resource: 0, road: false, capital: 0, structure_level: 0
        };
        n
    ];
    for (i, t) in tiles.iter_mut().enumerate() {
        t.idx = i;
    }
    if let Some(arr) = vs["tiles"].as_array() {
        for row in arr {
            let ti = row["tile"].as_u64().unwrap_or(0) as usize;
            if ti < n {
                let known = row["known"].as_bool().unwrap_or(false);
                tiles[ti].known = known;
                if known {
                    tiles[ti].terrain = row["terrain"].as_i64().unwrap_or(0) as i32;
                    tiles[ti].owner = row["owner"].as_i64().unwrap_or(0) as i32;
                    tiles[ti].structure = row["structure"].as_i64().unwrap_or(0) as i32;
                    tiles[ti].resource = row["resource"].as_i64().unwrap_or(0) as i32;
                    tiles[ti].road = row["road"].as_i64().unwrap_or(0) != 0;
                    tiles[ti].capital = row["capital"].as_i64().unwrap_or(0) as i32;
                    tiles[ti].structure_level = row["structure_level"].as_i64().unwrap_or(0) as i32;
                }
            }
        }
    }
    let mut units = Vec::new();
    let mut unit_at = vec![None; n];
    if let Some(arr) = vs["units"].as_array() {
        for row in arr {
            let tile = row["tile"].as_u64().unwrap_or(0) as usize;
            let u = Unit {
                tile,
                owner: row["owner"].as_i64().unwrap_or(0) as i32,
                kind: row["kind"].as_i64().unwrap_or(0) as i32,
                health: row["health"].as_f64().unwrap_or(10.0) as f32,
                veteran: row["veteran"].as_bool().unwrap_or(false),
                moved: row["moved"].as_bool().unwrap_or(false),
                attacked: row["attacked"].as_bool().unwrap_or(false),
                kills: row["kills"].as_i64().unwrap_or(0) as i32,
                home: row["home"].as_i64().unwrap_or(0) as i32,
                passenger: row["passenger"].as_i64().unwrap_or(0) as i32,
                poison: row["poison"].as_bool().unwrap_or(false),
                frozen: row["frozen"].as_bool().unwrap_or(false),
                boosted: row["boosted"].as_bool().unwrap_or(false),
                invisible: row["invisible"].as_bool().unwrap_or(false),
                detect: row["detect"].as_bool().unwrap_or(false),
            };
            if tile < n {
                unit_at[tile] = Some(units.len());
            }
            units.push(u);
        }
    }
    let mut cities = Vec::new();
    let mut city_at = vec![None; n];
    if let Some(arr) = vs["cities"].as_array() {
        for row in arr {
            let tile = row["tile"].as_u64().unwrap_or(0) as usize;
            let mut rewards = [0i32; 9];
            if let Some(ra) = row["rewards"].as_array() {
                for (i, rv) in ra.iter().enumerate().take(9) {
                    rewards[i] = rv.as_i64().unwrap_or(0) as i32;
                }
            }
            let c = City {
                tile,
                owner: row["owner"].as_i64().unwrap_or(0) as i32,
                level: row["level"].as_i64().unwrap_or(1) as i32,
                population: row["population"].as_i64().unwrap_or(0) as i32,
                production: row["production"].as_i64().unwrap_or(1) as i32,
                progress: row["progress"].as_i64().unwrap_or(0) as i32,
                connected: row["connected"].as_bool().unwrap_or(false),
                riot: row["riot"].as_bool().unwrap_or(false),
                rewards,
            };
            if tile < n {
                city_at[tile] = Some(cities.len());
            }
            cities.push(c);
        }
    }
    let e_idx = (enemy - 1) as usize;
    let rel = &vs["relations"];
    let get_rel = |k: &str| -> bool {
        rel[k].as_array().and_then(|a| a.get(e_idx)).and_then(|x| x.as_bool()).unwrap_or(false)
    };
    let mut legal = Vec::new();
    if let Some(arr) = v["legal_actions"].as_array() {
        for row in arr {
            let id = row["action_id"].as_str().unwrap_or("").to_owned();
            let c = &row["command"];
            legal.push(Legal {
                id,
                cmd: Cmd {
                    kind: c["kind"].as_i64().unwrap_or(0) as i32,
                    source: c["source"].as_i64().unwrap_or(-1) as i32,
                    target: c["target"].as_i64().unwrap_or(-1) as i32,
                    parameter: c["parameter"].as_i64().unwrap_or(0) as i32,
                },
            });
        }
    }
    Obs {
        player,
        enemy,
        state_version,
        size,
        n,
        turn,
        stars,
        tech,
        tiles,
        units,
        unit_at,
        cities,
        city_at,
        peace: get_rel("peace"),
        peace_offered_to_me: get_rel("peace_offers"),
        embassy_sent: get_rel("embassies_sent"),
        embassy_recv: get_rel("embassies_received"),
        known_enemy: get_rel("known_players"),
        legal,
    }
}
#[derive(Clone)]
struct MapInfo {
    my_cities: Vec<usize>,
    enemy_cities: Vec<usize>,
    neutral_villages: Vec<usize>,
    ruins: Vec<usize>,
    starfish: Vec<usize>,
    city_territory: HashMap<usize, Vec<usize>>,
    tile_to_my_city: Vec<Option<usize>>,
    known_count: usize,
    water_count: usize,
    ocean_count: usize,
    land_frontier_count: usize, // unknown tiles adjacent to reachable known land
    water_frontier_count: usize,
    needs_navy: bool,
    urgent_navy: bool,
    has_port: bool,
    my_ports: Vec<usize>,
    can_build_port_now: bool,
    has_shallow_water_in_territory: bool,
    enemy_target_center: usize,
    besieged_my_cities: Vec<usize>,
}

fn analyze_map(obs: &Obs, mem: &BotMemory) -> MapInfo {
    let n = obs.n;
    let size = obs.size;
    let mut my_cities = Vec::new();
    let mut enemy_cities = Vec::new();
    let mut neutral_villages = Vec::new();
    let mut ruins = Vec::new();
    let mut starfish = Vec::new();

    for c in &obs.cities {
        if c.owner == obs.player {
            my_cities.push(c.tile);
        } else if c.owner == obs.enemy {
            enemy_cities.push(c.tile);
        } else {
            neutral_villages.push(c.tile);
        }
    }
    // Also check remembered enemy cities / villages on tiles if not in cities list
    for (&ti, &st) in &mem.remembered_structures {
        if st == 1 && obs.city_at[ti].is_none() {
            let ow = mem.remembered_owners.get(&ti).copied().unwrap_or(0);
            if ow == obs.enemy {
                enemy_cities.push(ti);
            } else if ow == 0 {
                neutral_villages.push(ti);
            }
        }
    }

    let mut known_count = 0;
    let mut water_count = 0;
    let mut ocean_count = 0;
    let mut my_ports = Vec::new();

    for t in &obs.tiles {
        if t.known {
            known_count += 1;
            if t.terrain == 1 || t.terrain == 2 {
                water_count += 1;
                if t.terrain == 2 {
                    ocean_count += 1;
                }
            }
            if t.structure == 2 {
                ruins.push(t.idx);
            }
            if t.resource == 8 && t.structure == 0 {
                starfish.push(t.idx);
            }
            if t.structure == 8 && t.owner == obs.player {
                my_ports.push(t.idx);
            }
        }
    }

    // Approximate territory of each of our cities:
    // A city has border radius 1, or 2 if it took Border Growth (rewards[5] > 0).
    // Sort my_cities by level/capital to assign owned tiles.
    let mut city_territory: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut tile_to_my_city = vec![None; n];
    for &ct in &my_cities {
        city_territory.insert(ct, Vec::new());
    }
    for t in &obs.tiles {
        if t.known && t.owner == obs.player {
            // Find which of my_cities owns this tile
            let mut best_c = None;
            let mut best_d = 99;
            for &ct in &my_cities {
                let ci = obs.city_at[ct].unwrap();
                let rad = if obs.cities[ci].rewards[5] > 0 { 2 } else { 1 };
                let d = cheb(size, ct, t.idx);
                if d <= rad && d < best_d {
                    best_d = d;
                    best_c = Some(ct);
                }
            }
            if best_c.is_none() {
                for &ct in &my_cities {
                    let d = cheb(size, ct, t.idx);
                    if d <= 2 && d < best_d {
                        best_d = d;
                        best_c = Some(ct);
                    }
                }
            }
            if let Some(ct) = best_c {
                tile_to_my_city[t.idx] = Some(ct);
                city_territory.get_mut(&ct).unwrap().push(t.idx);
            }
        }
    }

    // BFS on known land from our cities & land units to see what is reachable by land
    let mut land_vis = vec![false; n];
    let mut q = VecDeque::new();
    for &ct in &my_cities {
        land_vis[ct] = true;
        q.push_back(ct);
    }
    for u in &obs.units {
        if u.owner == obs.player && !unit_stats(u.kind).float {
            if !land_vis[u.tile] {
                land_vis[u.tile] = true;
                q.push_back(u.tile);
            }
        }
    }
    while let Some(cur) = q.pop_front() {
        for nb in neighbors(size, cur) {
            if land_vis[nb] {
                continue;
            }
            let nt = &obs.tiles[nb];
            if nt.known && (nt.terrain >= 3 || nt.structure == 48) {
                // If mountain and we don't have Climbing, still part of landmass if we could research Climbing,
                // but for immediate walkability let's include all land >= 3 so we know if water blocks us
                land_vis[nb] = true;
                q.push_back(nb);
            }
        }
    }

    let mut land_frontier_count = 0;
    let mut water_frontier_count = 0;
    let mut land_reachable_villages = 0;
    let mut land_reachable_enemies = 0;

    for i in 0..n {
        if land_vis[i] {
            if obs.tiles[i].structure == 1 && obs.tiles[i].owner != obs.player {
                if obs.tiles[i].owner == 0 {
                    land_reachable_villages += 1;
                } else {
                    land_reachable_enemies += 1;
                }
            }
            if let Some(ui) = obs.unit_at[i] {
                if obs.units[ui].owner == obs.enemy {
                    land_reachable_enemies += 1;
                }
            }
            for nb in neighbors(size, i) {
                if !obs.tiles[nb].known {
                    land_frontier_count += 1;
                }
            }
        } else if obs.tiles[i].known && (obs.tiles[i].terrain == 1 || obs.tiles[i].terrain == 2) {
            for nb in neighbors(size, i) {
                if !obs.tiles[nb].known {
                    water_frontier_count += 1;
                }
            }
        }
    }

    let mut has_shallow_water_in_territory = false;
    let mut can_build_port_now = false;
    for t in &obs.tiles {
        if t.known && t.owner == obs.player && t.terrain == 1 && t.structure == 0 {
            let unit_ok = match obs.unit_at[t.idx] {
                None => true,
                Some(ui) => obs.units[ui].owner == obs.player,
            };
            if unit_ok {
                has_shallow_water_in_territory = true;
                if obs.tech[10] && obs.stars >= 7 {
                    can_build_port_now = true;
                }
            }
        }
    }

    let has_port = !my_ports.is_empty();
    let water_ratio = if known_count > 0 { (water_count as f32) / (known_count as f32) } else { 0.0 };
    let significant_water = water_count >= 14 || (ocean_count >= 3 && water_ratio >= 0.20);
    let island_sealed = significant_water && land_frontier_count == 0 && land_reachable_enemies == 0 && land_reachable_villages == 0;
    let island_nearly_done = significant_water && land_frontier_count <= 3 && land_reachable_enemies == 0 && land_reachable_villages == 0;
    let needs_navy = island_sealed
        || island_nearly_done
        || (ocean_count >= 4 && water_ratio >= 0.25)
        || (water_count >= 24 && water_ratio >= 0.30);
    let urgent_navy = island_sealed || (island_nearly_done && ocean_count >= 2);

    // Estimate enemy capital / target center if not yet known: mirror our capital across center of board!
    let enemy_target_center = if let Some(ec) = mem.enemy_capital {
        ec
    } else if let Some(&ec) = enemy_cities.first() {
        ec
    } else if let Some(mc) = mem.my_capital.or_else(|| my_cities.first().copied()) {
        let (mx, my) = (mc % size, mc / size);
        let (tx, ty) = ((size - 1).saturating_sub(mx), (size - 1).saturating_sub(my));
        ty * size + tx
    } else {
        (size / 2) * size + (size / 2)
    };

    let mut besieged_my_cities = Vec::new();
    for &ct in &my_cities {
        if let Some(ui) = obs.unit_at[ct] {
            if obs.units[ui].owner == obs.enemy {
                besieged_my_cities.push(ct);
            }
        }
    }

    MapInfo {
        my_cities,
        enemy_cities,
        neutral_villages,
        ruins,
        starfish,
        city_territory,
        tile_to_my_city,
        known_count,
        water_count,
        ocean_count,
        land_frontier_count,
        water_frontier_count,
        needs_navy,
        urgent_navy,
        has_port,
        my_ports,
        can_build_port_now,
        has_shallow_water_in_territory,
        enemy_target_center,
        besieged_my_cities,
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
struct Node {
    cost: i32,
    tile: usize,
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost).then_with(|| self.tile.cmp(&other.tile))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Multi-source or single-source shortest path cost (in half-movement steps, where 1 road step = 1, normal step = 2,
/// terminal forest/mountain step = 3, water/port transition = 4) for a unit at `start` (or generic land/naval).
fn shortest_distances_from_targets(obs: &Obs, targets: &[usize], for_naval: bool, allow_water_cross: bool) -> Vec<i32> {
    let n = obs.n;
    let size = obs.size;
    let mut dist = vec![1_000_000i32; n];
    let mut pq = BinaryHeap::new();
    for &t in targets {
        if t < n {
            dist[t] = 0;
            pq.push(Node { cost: 0, tile: t });
        }
    }
    let has_climbing = obs.tech[20];
    let has_fishing = obs.tech[10];
    let has_sailing = obs.tech[13];

    while let Some(Node { cost, tile }) = pq.pop() {
        if cost > dist[tile] {
            continue;
        }
        for nb in neighbors(size, tile) {
            let nt = &obs.tiles[nb];
            let step_cost = if !nt.known {
                3
            } else if for_naval {
                if nt.terrain == 1 {
                    2
                } else if nt.terrain == 2 {
                    if has_sailing || allow_water_cross { 2 } else { 50 }
                } else if nt.terrain >= 3 {
                    // Boats can step onto land to disembark, but cannot traverse across land as a boat
                    if tile == nb { 2 } else { 15 }
                } else {
                    1000
                }
            } else {
                // Land unit pathfinding backwards from target
                if nt.terrain == 3 {
                    if nt.road || nt.structure == 1 { 1 } else { 2 }
                } else if nt.terrain == 5 {
                    if nt.road { 1 } else { 3 }
                } else if nt.terrain == 4 {
                    if has_climbing { 4 } else { 80 }
                } else if nt.structure == 48 {
                    1
                } else if nt.structure == 8 && nt.owner == obs.player {
                    3
                } else if nt.terrain == 1 || nt.terrain == 2 {
                    if allow_water_cross && (has_fishing || nt.terrain == 1) {
                        if nt.terrain == 1 { 6 } else if has_sailing { 6 } else { 25 }
                    } else {
                        1000
                    }
                } else {
                    1000
                }
            };
            if step_cost < 500 {
                let nc = cost + step_cost;
                if nc < dist[nb] {
                    dist[nb] = nc;
                    pq.push(Node { cost: nc, tile: nb });
                }
            }
        }
    }
    dist
}

fn combat_formula(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    (attack, defense, (attack * 0.5).floor())
}

fn calc_defense_bonus(obs: &Obs, mem: &BotMemory, target_tile: usize, def_unit: &Unit) -> f32 {
    let t = &obs.tiles[target_tile];
    let dst_stats = unit_stats(def_unit.kind);
    let mut bonus = 1.0f32;
    if t.structure == 1 && t.owner == def_unit.owner && dst_stats.fortify {
        if def_unit.owner == obs.player {
            if let Some(ci) = obs.city_at[target_tile] {
                bonus = if obs.cities[ci].rewards[1] > 0 { 4.0 } else { 1.5 };
            } else {
                bonus = 1.5;
            }
        } else if mem.known_walled_cities.contains(&target_tile) {
            bonus = 4.0;
        } else {
            bonus = 1.5;
        }
    } else if t.terrain == 1 || t.terrain == 2 {
        if def_unit.owner == obs.player && obs.tech[12] {
            bonus = 1.5;
        }
    } else if t.terrain == 5 {
        if def_unit.owner == obs.player && obs.tech[18] {
            bonus = 1.5;
        } else if def_unit.owner == obs.enemy {
            // Assume enemy might have Archery if we saw Archers or turn >= 12
            bonus = 1.5;
        }
    } else if t.terrain == 4 {
        bonus = 1.5;
    }
    if def_unit.poison {
        bonus *= 0.5;
    }
    bonus
}

#[derive(Clone, Debug)]
struct AttackEval {
    dmg: f32,
    retal: f32,
    splash: f32,
    kills: bool,
    dies: bool,
    advances: bool,
    persist_chain: bool,
    score: f32,
}

fn evaluate_attack_from(obs: &Obs, mem: &BotMemory, att: &Unit, from_tile: usize, dst_tile: usize, info: &MapInfo) -> AttackEval {
    let a_stats = unit_stats(att.kind);
    // Cloak infiltration on enemy city
    if a_stats.infiltrate {
        let city_lvl = (obs.tiles[dst_tile].structure_level + 1).max(1);
        return AttackEval {
            dmg: 5.0,
            retal: 0.0,
            splash: 0.0,
            kills: false,
            dies: true,
            advances: false,
            persist_chain: false,
            score: 1200.0 + (city_lvl as f32) * 150.0,
        };
    }
    let def_idx = match obs.unit_at[dst_tile] {
        Some(idx) => idx,
        None => {
            return AttackEval {
                dmg: 0.0, retal: 0.0, splash: 0.0, kills: false, dies: false, advances: false, persist_chain: false, score: -1e9,
            };
        }
    };
    let def = &obs.units[def_idx];
    // MindBender conversion
    if att.kind == 10 {
        let val = unit_star_value(def);
        return AttackEval {
            dmg: def.health,
            retal: 0.0,
            splash: 0.0,
            kills: true,
            dies: false,
            advances: false,
            persist_chain: false,
            score: 900.0 + val * 80.0 + def.health * 15.0,
        };
    }

    let d_stats = unit_stats(def.kind);
    let am = unit_max_hp(att);
    let dm = unit_max_hp(def);
    let db = calc_defense_bonus(obs, mem, dst_tile, def);
    let dd = d_stats.defense * if def.poison { 0.7 } else { 1.0 };
    let (dmg, raw_retal, splash) = combat_formula(a_stats.attack, att.health, am, dd, def.health, dm, db);
    let kills = dmg >= def.health - 1e-4;
    let dist = cheb(obs.size, from_tile, dst_tile);
    let can_retal = !kills && !a_stats.surprise && !d_stats.stiff && dist <= d_stats.range;
    let retal = if can_retal { raw_retal } else { 0.0 };
    let dies = !kills && retal >= att.health - 1e-4;

    let dt = &obs.tiles[dst_tile];
    let can_enter = dt.terrain == 3
        || dt.terrain == 5
        || (dt.terrain == 4 && obs.tech[20])
        || (dt.terrain <= 2 && (a_stats.float || dt.structure == 48 || (dt.structure == 8 && dt.owner == obs.player)));
    let advances = kills && !dies && a_stats.range < 2 && can_enter;

    // Check if Knight Persist has another target from dst_tile
    let mut persist_chain = false;
    if kills && a_stats.persist && advances {
        for nb in neighbors(obs.size, dst_tile) {
            if nb != from_tile {
                if let Some(ui) = obs.unit_at[nb] {
                    if obs.units[ui].owner == obs.enemy {
                        persist_chain = true;
                        break;
                    }
                }
            }
        }
    }

    // Calculate splash damage dealt to adjacent enemies (Bomber = 41)
    let mut total_splash_dmg = 0.0f32;
    let mut splash_kills = 0;
    if a_stats.splash {
        for nb in neighbors(obs.size, dst_tile) {
            if let Some(ui) = obs.unit_at[nb] {
                let eu = &obs.units[ui];
                if eu.owner == obs.enemy {
                    let ed_stats = unit_stats(eu.kind);
                    let edb = calc_defense_bonus(obs, mem, nb, eu);
                    let edd = ed_stats.defense * if eu.poison { 0.7 } else { 1.0 };
                    let (_, _, sp) = combat_formula(a_stats.attack, att.health, am, edd, eu.health, unit_max_hp(eu), edb);
                    let actual = sp.min(eu.health);
                    total_splash_dmg += actual;
                    if sp >= eu.health - 1e-4 {
                        splash_kills += 1;
                    }
                }
            }
        }
    }

    let def_val = unit_star_value(def);
    let att_val = unit_star_value(att);
    let actual_dmg = dmg.min(def.health);
    let actual_retal = retal.min(att.health);

    let mut score = 0.0f32;
    // Value of damage dealt vs retaliation taken
    score += (actual_dmg / dm.max(10.0)) * def_val * 35.0;
    score += total_splash_dmg * 18.0 + (splash_kills as f32) * 140.0;
    score -= (actual_retal / am.max(10.0)) * att_val * 28.0;

    if dies {
        score -= att_val * 85.0 + 250.0;
    }

    let on_my_city = info.besieged_my_cities.contains(&dst_tile);
    let on_enemy_city = dt.structure == 1 && dt.owner == obs.enemy;
    let on_neutral_village = dt.structure == 1 && dt.owner == 0;

    if on_my_city {
        // Breaking siege on our own city is top priority!
        score += actual_dmg * 45.0;
        if kills {
            score += 950.0;
        }
    }

    if kills {
        score += 180.0 + def_val * 25.0;
        if persist_chain {
            score += 380.0;
        } else if a_stats.persist {
            score += 160.0;
        }
        if on_enemy_city {
            if advances {
                // Melee unit killing defender and stepping onto enemy city!
                score += 850.0;
                // Bonus if attacker has enough remaining HP to survive next turn and capture
                score += att.health * 12.0;
            } else {
                // Ranged unit killing defender on enemy city: great if we have another unit that can step onto dst_tile,
                // or still good to clear the city, EXCEPT if we have a melee unit that ALREADY moved adjacent to dst_tile
                // and can kill-advance onto dst_tile while no unmoved unit can step onto dst_tile!
                let mut unmoved_can_step = false;
                let mut moved_melee_can_kill_advance = false;
                for u in &obs.units {
                    if u.owner == obs.player && u.tile != att.tile {
                        let us = unit_stats(u.kind);
                        if !u.moved && cheb(obs.size, u.tile, dst_tile) <= us.movement {
                            unmoved_can_step = true;
                        }
                        if !u.attacked && us.range == 1 && cheb(obs.size, u.tile, dst_tile) == 1 {
                            let (mdmg, _, _) = combat_formula(us.attack, u.health, unit_max_hp(u), dd, def.health, dm, db);
                            if mdmg >= def.health - 1e-4 {
                                moved_melee_can_kill_advance = true;
                            }
                        }
                    }
                }
                if moved_melee_can_kill_advance && !unmoved_can_step {
                    // Let the adjacent melee unit get the kill so it advances onto the enemy city!
                    score -= 500.0;
                } else {
                    score += 320.0;
                }
            }
        } else if on_neutral_village && advances {
            score += 450.0;
        }
    } else {
        // Non-lethal attack:
        // Ranged / non-retaliated attacks MUST go before melee attacks so they soften targets!
        if retal == 0.0 {
            score += 95.0;
            if on_enemy_city {
                score += 120.0;
            }
        } else {
            // Retaliated non-lethal melee attack: check if we have other attackers that can help finish `def` this turn!
            let mut ally_followup_dmg = 0.0f32;
            for u in &obs.units {
                if u.owner == obs.player && u.tile != att.tile && !u.attacked {
                    let us = unit_stats(u.kind);
                    let d = cheb(obs.size, u.tile, dst_tile);
                    if d <= us.range || (!u.moved && us.dash && d <= us.movement + us.range) {
                        let (admg, _, _) = combat_formula(us.attack, u.health, unit_max_hp(u), dd, (def.health - dmg).max(1.0), dm, db);
                        ally_followup_dmg += admg;
                    }
                }
            }
            if dmg + ally_followup_dmg >= def.health - 1e-4 {
                // Combined focus fire can kill this enemy this turn!
                score += 140.0;
                if on_enemy_city || on_my_city {
                    score += 260.0;
                }
            } else if actual_retal > actual_dmg * 1.1 && !on_my_city {
                // Bad trade with no kill follow-up
                score -= 220.0;
            }
        }
    }

    // Never vacate our own city via melee advance if an enemy unit is adjacent and could step onto our city!
    if advances && obs.tiles[from_tile].structure == 1 && obs.tiles[from_tile].owner == obs.player {
        let enemy_threat = obs.units.iter().any(|eu| {
            eu.owner == obs.enemy
                && eu.tile != dst_tile
                && cheb(obs.size, eu.tile, from_tile) <= unit_stats(eu.kind).movement.max(1) + 1
        });
        if enemy_threat {
            score -= 600.0;
        }
    }

    // Bonus for getting the 3rd kill for veterancy promotion
    if kills && !att.veteran && !a_stats.stat && !a_stats.carry && att.kills == 2 {
        score += 110.0;
    }

    AttackEval {
        dmg,
        retal,
        splash,
        kills,
        dies,
        advances,
        persist_chain,
        score,
    }
}
#[derive(Clone, Debug)]
struct EconAction {
    action_idx: usize,
    city_tile: usize,
    cost: i32,
    pop_gain: i32,
    stars_per_turn_gain: i32,
    is_port: bool,
    is_monument: bool,
    is_road: bool,
}

/// Evaluates all legal Harvest (kind 5) and Build (kind 6) actions to see how much population and income they give.
fn gather_econ_actions(obs: &Obs, info: &MapInfo) -> Vec<EconAction> {
    let mut out = Vec::new();
    let size = obs.size;

    for (idx, leg) in obs.legal.iter().enumerate() {
        let c = leg.cmd;
        if c.kind == 5 {
            // Harvest resource on target tile (cost 2, +1 pop)
            let dst = c.target as usize;
            if let Some(ct) = info.tile_to_my_city[dst] {
                out.push(EconAction {
                    action_idx: idx,
                    city_tile: ct,
                    cost: 2,
                    pop_gain: 1,
                    stars_per_turn_gain: 0,
                    is_port: false,
                    is_monument: false,
                    is_road: false,
                });
            }
        } else if c.kind == 6 {
            let dst = c.target as usize;
            let st = c.parameter;
            let city_opt = info.tile_to_my_city[dst];
            match st {
                12 => {
                    // LumberHut: cost 3, +1 pop (+1 more if adjacent to our Sawmill!)
                    if let Some(ct) = city_opt {
                        let mut pop = 1;
                        let mut spt = 0;
                        for nb in neighbors(size, dst) {
                            if obs.tiles[nb].owner == obs.player && obs.tiles[nb].structure == 13 {
                                if let Some(saw_ct) = info.tile_to_my_city[nb] {
                                    if saw_ct == ct {
                                        pop += 1;
                                    }
                                }
                                // Also check if a Market is adjacent to that Sawmill!
                                for mnb in neighbors(size, nb) {
                                    if obs.tiles[mnb].owner == obs.player && obs.tiles[mnb].structure == 50 && obs.tiles[mnb].structure_level < 8 {
                                        spt += 1;
                                    }
                                }
                            }
                        }
                        out.push(EconAction {
                            action_idx: idx, city_tile: ct, cost: 3, pop_gain: pop, stars_per_turn_gain: spt,
                            is_port: false, is_monument: false, is_road: false,
                        });
                    }
                }
                13 => {
                    // Sawmill: cost 5, +1 pop per adjacent owned LumberHut (12)
                    if let Some(ct) = city_opt {
                        let mut adj_huts = 0;
                        let mut adj_forests = 0;
                        for nb in neighbors(size, dst) {
                            if obs.tiles[nb].owner == obs.player {
                                if obs.tiles[nb].structure == 12 {
                                    adj_huts += 1;
                                } else if obs.tiles[nb].terrain == 5 && obs.tiles[nb].structure == 0 {
                                    adj_forests += 1;
                                }
                            }
                        }
                        // Only consider Sawmill placements that have at least 1 hut (or 2+ huts/forests)
                        if adj_huts >= 1 || adj_forests >= 2 {
                            out.push(EconAction {
                                action_idx: idx, city_tile: ct, cost: 5, pop_gain: adj_huts, stars_per_turn_gain: 0,
                                is_port: false, is_monument: false, is_road: false,
                            });
                        }
                    }
                }
                5 => {
                    // Farm: cost 5, +2 pop (+1 if adjacent to owned Windmill 6)
                    if let Some(ct) = city_opt {
                        let mut pop = 2;
                        for nb in neighbors(size, dst) {
                            if obs.tiles[nb].owner == obs.player && obs.tiles[nb].structure == 6 {
                                if info.tile_to_my_city[nb] == Some(ct) {
                                    pop += 1;
                                }
                            }
                        }
                        out.push(EconAction {
                            action_idx: idx, city_tile: ct, cost: 5, pop_gain: pop, stars_per_turn_gain: 0,
                            is_port: false, is_monument: false, is_road: false,
                        });
                    }
                }
                6 => {
                    // Windmill: cost 5, +1 pop per adjacent owned Farm (5)
                    if let Some(ct) = city_opt {
                        let adj_farms = neighbors(size, dst)
                            .into_iter()
                            .filter(|&nb| obs.tiles[nb].owner == obs.player && obs.tiles[nb].structure == 5)
                            .count() as i32;
                        if adj_farms >= 1 {
                            out.push(EconAction {
                                action_idx: idx, city_tile: ct, cost: 5, pop_gain: adj_farms, stars_per_turn_gain: 0,
                                is_port: false, is_monument: false, is_road: false,
                            });
                        }
                    }
                }
                21 => {
                    // Mine: cost 5, +2 pop (+2 if adjacent to owned Forge 22)
                    if let Some(ct) = city_opt {
                        let mut pop = 2;
                        for nb in neighbors(size, dst) {
                            if obs.tiles[nb].owner == obs.player && obs.tiles[nb].structure == 22 {
                                if info.tile_to_my_city[nb] == Some(ct) {
                                    pop += 2;
                                }
                            }
                        }
                        out.push(EconAction {
                            action_idx: idx, city_tile: ct, cost: 5, pop_gain: pop, stars_per_turn_gain: 0,
                            is_port: false, is_monument: false, is_road: false,
                        });
                    }
                }
                22 => {
                    // Forge: cost 5, +2 pop per adjacent owned Mine (21)
                    if let Some(ct) = city_opt {
                        let adj_mines = neighbors(size, dst)
                            .into_iter()
                            .filter(|&nb| obs.tiles[nb].owner == obs.player && obs.tiles[nb].structure == 21)
                            .count() as i32;
                        if adj_mines >= 1 {
                            out.push(EconAction {
                                action_idx: idx, city_tile: ct, cost: 5, pop_gain: adj_mines * 2, stars_per_turn_gain: 0,
                                is_port: false, is_monument: false, is_road: false,
                            });
                        }
                    }
                }
                8 => {
                    // Port: cost 7, +1 pop (+1 more if connects disconnected city to capital!)
                    if let Some(ct) = city_opt {
                        let mut pop = 1;
                        if let Some(ci) = obs.city_at[ct] {
                            if !obs.cities[ci].connected && !info.my_ports.is_empty() {
                                if info.my_ports.iter().any(|&p| cheb(size, p, dst) <= 4) {
                                    pop += 1;
                                }
                            }
                        }
                        out.push(EconAction {
                            action_idx: idx, city_tile: ct, cost: 7, pop_gain: pop, stars_per_turn_gain: 0,
                            is_port: true, is_monument: false, is_road: false,
                        });
                    }
                }
                50 => {
                    // Market: cost 5, +1 star/turn per adjacent Sawmill(13)/Windmill(6)/Forge(22) level!
                    if let Some(ct) = city_opt {
                        let mut spt = 0;
                        for nb in neighbors(size, dst) {
                            if obs.tiles[nb].owner == obs.player && matches!(obs.tiles[nb].structure, 6 | 13 | 22) {
                                spt += obs.tiles[nb].structure_level.max(1);
                            }
                        }
                        if spt >= 1 {
                            out.push(EconAction {
                                action_idx: idx, city_tile: ct, cost: 5, pop_gain: 0, stars_per_turn_gain: spt.min(8),
                                is_port: false, is_monument: false, is_road: false,
                            });
                        }
                    }
                }
                23..=29 => {
                    // Monument: cost 0, +3 pop!
                    if let Some(ct) = city_opt {
                        out.push(EconAction {
                            action_idx: idx, city_tile: ct, cost: 0, pop_gain: 3, stars_per_turn_gain: 0,
                            is_port: false, is_monument: true, is_road: false,
                        });
                    }
                }
                _ => {}
            }
        }
    }
    out
}

/// Checks if building a single road/bridge tile connects one of our unconnected cities to our connected network!
fn find_one_step_city_connection_road(obs: &Obs, mem: &BotMemory) -> Option<(usize, usize)> {
    // Returns (action_idx, connected_city_tile)
    let cap = mem.my_capital.or_else(|| obs.cities.iter().find(|c| c.owner == obs.player).map(|c| c.tile))?;
    let size = obs.size;
    let n = obs.n;

    // Find all tiles currently connected to `cap` via friendly roads/cities/bridges/ports
    let mut connected_net = vec![false; n];
    let mut q = VecDeque::new();
    connected_net[cap] = true;
    q.push_back(cap);
    while let Some(cur) = q.pop_front() {
        for nb in neighbors(size, cur) {
            if connected_net[nb] {
                continue;
            }
            let t = &obs.tiles[nb];
            if t.known && (t.owner == 0 || t.owner == obs.player) {
                let is_node = t.road || t.structure == 48 || t.structure == 8 || (t.structure == 1 && t.owner == obs.player);
                if is_node {
                    connected_net[nb] = true;
                    q.push_back(nb);
                }
            }
        }
    }

    // For any unconnected owned city, check if a single road/bridge tile (cost 3 or 5) bridges `connected_net` to that city!
    for c in &obs.cities {
        if c.owner == obs.player && c.tile != cap && !c.connected {
            for (idx, leg) in obs.legal.iter().enumerate() {
                if leg.cmd.kind == 6 && (leg.cmd.parameter == 3 || leg.cmd.parameter == 48) {
                    let rt = leg.cmd.target as usize;
                    let adj_to_net = neighbors(size, rt).into_iter().any(|nb| connected_net[nb]);
                    let adj_to_city = cheb(size, rt, c.tile) == 1;
                    if adj_to_net && adj_to_city {
                        return Some((idx, c.tile));
                    }
                }
            }
        }
    }
    None
}
/// Compute true shortest-path distance from a single target tile `target` to all tiles on the board.
fn bfs_dist_from_target(obs: &Obs, target: usize, for_naval: bool, allow_water: bool) -> Vec<i32> {
    shortest_distances_from_targets(obs, &[target], for_naval, allow_water)
}

/// Assign each movable friendly unit a distinct strategic target (village, ruin, starfish, port, enemy city, or exploration frontier)
/// so multiple units spread out and never clump onto the same village or frontier!
fn assign_unit_targets(obs: &Obs, info: &MapInfo) -> HashMap<usize, usize> {
    let size = obs.size;
    let mut assignments: HashMap<usize, usize> = HashMap::new();
    let mut claimed_single_targets: HashSet<usize> = HashSet::new();

    // Any target that ALREADY has a friendly unit standing on it is already claimed!
    for u in &obs.units {
        if u.owner == obs.player {
            let st = obs.tiles[u.tile].structure;
            if (st == 1 && obs.tiles[u.tile].owner != obs.player) || st == 2 || obs.tiles[u.tile].resource == 8 {
                claimed_single_targets.insert(u.tile);
                assignments.insert(u.tile, u.tile);
            }
        }
    }

    let movable_units: Vec<&Unit> = obs
        .units
        .iter()
        .filter(|u| u.owner == obs.player && !u.moved && !assignments.contains_key(&u.tile))
        .collect();

    if movable_units.is_empty() {
        return assignments;
    }

    // Build candidate target list: (target_tile, is_single_unit_only, base_priority, naval_accessible_only)
    let mut targets: Vec<(usize, bool, f32)> = Vec::new();

    for &v in &info.neutral_villages {
        if !claimed_single_targets.contains(&v) {
            targets.push((v, true, 500.0));
        }
    }
    for &ec in &info.enemy_cities {
        // Enemy cities can take multiple units!
        targets.push((ec, false, 520.0));
    }
    for &r in &info.ruins {
        if !claimed_single_targets.contains(&r) {
            targets.push((r, true, 440.0));
        }
    }
    if obs.tech[14] {
        for &sf in &info.starfish {
            if !claimed_single_targets.contains(&sf) {
                targets.push((sf, true, 410.0));
            }
        }
    }
    if info.needs_navy {
        for &p in &info.my_ports {
            if obs.unit_at[p].is_none() && !claimed_single_targets.contains(&p) {
                targets.push((p, true, if info.urgent_navy { 460.0 } else { 340.0 }));
            }
        }
    }

    // Add well-spaced exploration frontier tiles (known tiles adjacent to unknown tiles)
    let mut frontier_candidates: Vec<(f32, usize)> = Vec::new();
    for t in &obs.tiles {
        if !t.known {
            continue;
        }
        let passable_land = t.terrain == 3 || t.terrain == 5 || (t.terrain == 4 && obs.tech[20]) || t.structure == 48;
        let passable_water = (t.terrain == 1 && obs.tech[10]) || (t.terrain == 2 && obs.tech[13]);
        if !passable_land && !passable_water {
            continue;
        }
        let mut unknown_nbs = 0;
        for nb in neighbors(size, t.idx) {
            if !obs.tiles[nb].known {
                unknown_nbs += 1;
            }
        }
        if unknown_nbs > 0 {
            let (x, y) = ((t.idx % size) as i32, (t.idx / size) as i32);
            let s = size as i32;
            let edge_dist = x.min(s - 1 - x).min(y.min(s - 1 - y));
            let center_tile = (size / 2) * size + (size / 2);
            let dist_center = cheb(size, t.idx, center_tile);
            let interior_bonus = if edge_dist >= 2 { 55.0 } else if edge_dist == 1 { 5.0 } else { -90.0 };
            let toward_enemy = 60.0 - (cheb(size, t.idx, info.enemy_target_center) as f32) * 4.5 - (dist_center as f32) * 3.5;
            let f_score = 260.0 + (unknown_nbs as f32) * 12.0 + interior_bonus + toward_enemy;
            frontier_candidates.push((f_score, t.idx));
        }
    }
    frontier_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    let mut chosen_frontiers: Vec<usize> = Vec::new();
    for (f_sc, ft) in frontier_candidates {
        if chosen_frontiers.iter().all(|&cf| cheb(size, cf, ft) >= 2) {
            chosen_frontiers.push(ft);
            targets.push((ft, true, f_sc));
            if chosen_frontiers.len() >= 16 {
                break;
            }
        }
    }

    // Always include enemy_target_center as a multi-unit fallback target
    targets.push((info.enemy_target_center, false, 240.0));

    // Precompute shortest-path distance maps for each target (land and naval)
    let mut land_dists: HashMap<usize, Vec<i32>> = HashMap::new();
    let mut naval_dists: HashMap<usize, Vec<i32>> = HashMap::new();
    for &(t, _, _) in &targets {
        land_dists.entry(t).or_insert_with(|| bfs_dist_from_target(obs, t, false, info.has_port));
        naval_dists.entry(t).or_insert_with(|| bfs_dist_from_target(obs, t, true, true));
    }

    // Greedily match (unit, target) pairs by highest (priority - distance_penalty)
    let mut unassigned: HashSet<usize> = movable_units.iter().map(|u| u.tile).collect();
    while !unassigned.is_empty() {
        let mut best_pair: Option<(f32, usize, usize, bool)> = None;
        for u in &movable_units {
            if !unassigned.contains(&u.tile) {
                continue;
            }
            let is_boat = unit_stats(u.kind).float;
            for &(targ, single_only, prio) in &targets {
                if single_only && claimed_single_targets.contains(&targ) {
                    continue;
                }
                // Don't assign a boat to our own port!
                if is_boat && info.my_ports.contains(&targ) {
                    continue;
                }
                let d = if is_boat {
                    naval_dists[&targ][u.tile]
                } else {
                    land_dists[&targ][u.tile]
                };
                if d >= 500_000 {
                    continue;
                }
                let val = prio - (d as f32) * 18.0;
                if best_pair.as_ref().map_or(true, |(b, _, _, _)| val > *b) {
                    best_pair = Some((val, u.tile, targ, single_only));
                }
            }
        }
        if let Some((_, utile, targ, single_only)) = best_pair {
            unassigned.remove(&utile);
            assignments.insert(utile, targ);
            if single_only {
                claimed_single_targets.insert(targ);
            }
        } else {
            break;
        }
    }

    assignments
}

fn choose(raw: &Value, mem: &mut BotMemory) -> String {
    let obs = parse_obs(raw);
    mem.update(&obs, raw);
    if obs.legal.is_empty() {
        return String::new();
    }
    if obs.legal.len() == 1 {
        return obs.legal[0].id.clone();
    }

    let info = analyze_map(&obs, mem);
    let size = obs.size;
    let num_my_cities = info.my_cities.len() as i32;

    // 1. MANDATORY CITY REWARDS (kind == 9)
    if obs.legal.iter().any(|l| l.cmd.kind == 9) {
        let mut best_id = obs.legal[0].id.clone();
        let mut best_score = -1e9f32;
        for leg in &obs.legal {
            if leg.cmd.kind != 9 {
                continue;
            }
            let ct = leg.cmd.target as usize;
            let param = leg.cmd.parameter;
            let s = match param {
                // Level 2: Workshop (3) vs Explorer (4)
                3 => 2000.0,
                4 => 1200.0,
                // Level 3: City Wall (1) vs Resources +5 stars (7)
                1 => {
                    let enemy_near = obs.units.iter().any(|u| u.owner == obs.enemy && cheb(size, u.tile, ct) <= 2);
                    if enemy_near { 2200.0 } else { 1500.0 }
                }
                7 => 2000.0,
                // Level 4: Border Growth (5) vs Population +3 (8)
                5 => {
                    let city_has_port_spot = info.city_territory.get(&ct).map_or(false, |tiles| {
                        tiles.iter().any(|&ti| obs.tiles[ti].terrain == 1)
                    });
                    let mut outer_water = 0;
                    let mut outer_value = 0;
                    for ti in 0..obs.n {
                        if cheb(size, ct, ti) == 2 && obs.tiles[ti].known && obs.tiles[ti].owner == 0 {
                            if obs.tiles[ti].terrain == 1 {
                                outer_water += 1;
                            }
                            if obs.tiles[ti].resource > 0 || obs.tiles[ti].terrain == 5 {
                                outer_value += 1;
                            }
                        }
                    }
                    let progress = obs.city_at[ct].map_or(0, |ci| obs.cities[ci].progress);
                    if progress >= 2 {
                        1400.0
                    } else if info.needs_navy && !info.has_port && !city_has_port_spot && outer_water > 0 {
                        2500.0
                    } else if outer_value >= 5 && progress == 0 {
                        2100.0
                    } else {
                        1700.0
                    }
                }
                8 => 2000.0,
                // Level 5+: Park (2) vs Giant (6)
                6 => 2500.0,
                2 => 1500.0,
                _ => 1000.0,
            };
            if s > best_score {
                best_score = s;
                best_id = leg.id.clone();
            }
        }
        return best_id;
    }

    // 2. FREE PROMOTIONS (kind == 3, param == 16)
    if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 3 && l.cmd.parameter == 16) {
        return leg.id.clone();
    }

    // 3. RUIN & STARFISH CAPTURES (kind == 8 on structure == 2 or resource == 8)
    for leg in &obs.legal {
        if leg.cmd.kind == 8 {
            let src = leg.cmd.source as usize;
            if obs.tiles[src].structure == 2 || obs.tiles[src].resource == 8 {
                return leg.id.clone();
            }
        }
    }

    let has_city_capture_ready = obs.legal.iter().any(|l| l.cmd.kind == 8 && obs.tiles[l.cmd.source as usize].structure == 1);

    // Count resources & terrain in our owned territory
    let mut owned_fruits = 0;
    let mut owned_fish = 0;
    let mut owned_forests = 0;
    let mut owned_huts = 0;
    let mut owned_crops = 0;
    let mut owned_metals = 0;
    let mut owned_mines = 0;
    let mut owned_farms = 0;
    let mut owned_shallow = 0;
    let mut max_sawmill_level = 0;

    for t in &obs.tiles {
        if t.known && t.owner == obs.player && t.structure == 0 {
            match t.resource {
                2 => owned_crops += 1,
                3 => owned_fish += 1,
                5 => owned_metals += 1,
                6 => owned_fruits += 1,
                _ => {}
            }
            if t.terrain == 5 {
                owned_forests += 1;
            }
            if t.terrain == 1 {
                owned_shallow += 1;
            }
        } else if t.known && t.owner == obs.player {
            if t.structure == 12 {
                owned_huts += 1;
            } else if t.structure == 21 {
                owned_mines += 1;
            } else if t.structure == 5 {
                owned_farms += 1;
            } else if matches!(t.structure, 6 | 13 | 22) {
                max_sawmill_level = max_sawmill_level.max(t.structure_level);
            }
        }
    }

    let my_unit_count = obs.units.iter().filter(|u| u.owner == obs.player).count();
    let has_empty_city_for_summon = obs.legal.iter().any(|l| l.cmd.kind == 4 && !matches!(l.cmd.parameter, 41 | 42 | 44));

    let eval_tech = |tech_id: usize| -> f32 {
        let cost = research_cost_of(&obs, tech_id, num_my_cities);
        let stars_after = obs.stars - cost;
        let pre_cap_bonus = if has_city_capture_ready { (tech_tier(tech_id) as f32) * 45.0 } else { 0.0 };
        let mut s = pre_cap_bonus - (cost as f32) * 3.5;

        match tech_id {
            16 => {
                // Forestry (LumberHut costs 3)
                if owned_forests == 0 {
                    return -1000.0;
                }
                if stars_after < 3 && !has_city_capture_ready && has_empty_city_for_summon && my_unit_count < (num_my_cities as usize + 2) {
                    return -500.0;
                }
                s += (owned_forests as f32) * 40.0 + if stars_after >= 3 { 110.0 } else { 20.0 };
            }
            1 => {
                // Riding (Rider costs 3)
                if info.urgent_navy {
                    return -200.0;
                }
                // Don't research Riding on Turn 1 with 0 stars left when we only have 1 unit! Train a 2nd Warrior first or wait until we have 8 stars!
                if my_unit_count <= 1 && stars_after < 2 && !has_city_capture_ready {
                    return -500.0;
                }
                s += 175.0 + if stars_after >= 3 { 95.0 } else { 0.0 };
            }
            6 => {
                // Organization (Fruit harvest costs 2)
                if owned_fruits == 0 && num_my_cities < 3 {
                    return -1000.0;
                }
                if stars_after < 2 && !has_city_capture_ready && has_empty_city_for_summon {
                    return -500.0;
                }
                s += (owned_fruits as f32) * 55.0 + if stars_after >= 2 { 85.0 } else { 10.0 };
            }
            10 => {
                // Fishing (Fish harvest costs 2, Port costs 7)
                if !info.needs_navy && owned_fish == 0 {
                    return -1000.0;
                }
                s += (owned_fish as f32) * 55.0;
                if info.urgent_navy {
                    s += 480.0;
                } else if info.needs_navy {
                    s += 270.0;
                } else if stars_after >= 2 && owned_fish > 0 {
                    s += 80.0;
                }
            }
            13 => {
                // Sailing (Ocean movement & Scoutship)
                if !info.needs_navy && !info.has_port {
                    return -1000.0;
                }
                if info.urgent_navy || info.has_port {
                    s += 500.0;
                } else {
                    s += 280.0;
                }
            }
            44 => {
                // Ramming (Rammership)
                if !info.needs_navy && !info.has_port {
                    return -1000.0;
                }
                if info.has_port {
                    s += 390.0;
                } else {
                    s += 200.0;
                }
            }
            14 => {
                // Navigation (Bomber & Starfish)
                if !info.needs_navy && !info.has_port && info.starfish.is_empty() {
                    return -1000.0;
                }
                let sf_bonus = (info.starfish.len() as f32) * 65.0;
                if info.has_port {
                    s += 340.0 + sf_bonus;
                } else {
                    s += 150.0 + sf_bonus;
                }
            }
            12 => {
                // Aquatism
                let my_boats = obs.units.iter().filter(|u| u.owner == obs.player && unit_stats(u.kind).float).count();
                if my_boats < 2 {
                    return -1000.0;
                }
                s += 240.0;
            }
            17 => {
                // Mathematics (Sawmill costs 5, Catapult costs 8)
                if owned_huts < 2 && obs.stars < cost + 8 {
                    return -500.0;
                }
                s += (owned_huts as f32) * 48.0 + if stars_after >= 5 { 135.0 } else { 40.0 };
            }
            4 => {
                // Roads (Road costs 3)
                if num_my_cities < 2 && obs.turn < 12 {
                    return -1000.0;
                }
                let unconn = obs.cities.iter().filter(|c| c.owner == obs.player && !c.connected).count();
                if unconn == 0 && max_sawmill_level < 2 && stars_after < 6 {
                    return -300.0;
                }
                s += (unconn as f32) * 65.0 + if stars_after >= 3 { 115.0 } else { 30.0 };
            }
            5 => {
                // Trade (Market costs 5)
                if max_sawmill_level < 2 {
                    return -1000.0;
                }
                s += (max_sawmill_level as f32) * 95.0 + if stars_after >= 5 { 140.0 } else { 30.0 };
            }
            18 => {
                // Archery (Archer costs 3 & 1.5x forest defense)
                if my_unit_count < 3 && !has_city_capture_ready {
                    return -300.0;
                }
                s += 150.0 + if stars_after >= 3 { 60.0 } else { 0.0 };
            }
            7 => {
                // Strategy (Defender costs 3)
                if my_unit_count < 3 && !info.needs_navy {
                    return -400.0;
                }
                if info.needs_navy || !info.enemy_cities.is_empty() {
                    s += 175.0 + if stars_after >= 3 { 50.0 } else { 0.0 };
                } else {
                    s += 90.0;
                }
            }
            38 => {
                // Diplomacy (Cloak costs 8 & capital vision)
                if num_my_cities < 3 && stars_after < 5 {
                    return -500.0;
                }
                s += if mem.enemy_capital.is_none() && obs.known_enemy { 210.0 } else { 145.0 };
            }
            20 => {
                // Climbing
                if num_my_cities < 2 && !info.urgent_navy && obs.turn < 8 {
                    return -500.0;
                }
                let mtn_in_terr = obs.tiles.iter().filter(|t| t.known && t.owner == obs.player && t.terrain == 4).count();
                if mtn_in_terr == 0 && obs.turn < 12 {
                    return -500.0;
                }
                s += (mtn_in_terr as f32) * 35.0 + 75.0;
            }
            23 => {
                // Mining (Mine costs 5)
                if owned_metals == 0 {
                    return -1000.0;
                }
                s += (owned_metals as f32) * 85.0 + if stars_after >= 5 { 90.0 } else { 10.0 };
            }
            24 => {
                // Smithery (Swordsman & Forge)
                s += (owned_mines as f32) * 75.0 + if stars_after >= 5 { 195.0 } else { 80.0 };
            }
            8 => {
                // Farming (Farm costs 5)
                if owned_crops == 0 {
                    return -1000.0;
                }
                s += (owned_crops as f32) * 85.0 + if stars_after >= 5 { 90.0 } else { 10.0 };
            }
            9 => {
                // Construction (Windmill costs 5)
                if owned_farms < 2 {
                    return -1000.0;
                }
                s += (owned_farms as f32) * 60.0 + 50.0;
            }
            2 => {
                // FreeSpirit (leads to Chivalry)
                if num_my_cities < 3 || obs.stars < cost + 6 {
                    return -500.0;
                }
                s += 130.0;
            }
            3 => {
                // Chivalry (Knight costs 8!)
                s += 280.0 + if stars_after >= 8 { 120.0 } else { 0.0 };
            }
            21 => {
                if num_my_cities < 4 || obs.stars < cost + 8 {
                    return -800.0;
                }
                s += 80.0;
            }
            22 => {
                s += 190.0;
            }
            _ => {
                return -1000.0;
            }
        }
        s
    };

    // Pre-capture tech discount: if we are about to capture a city AND can afford a genuinely useful tech right now
    if has_city_capture_ready && info.besieged_my_cities.is_empty() {
        let mut best_pre_tech: Option<(f32, String)> = None;
        for leg in &obs.legal {
            if leg.cmd.kind == 7 {
                let tid = leg.cmd.parameter as usize;
                let sc = eval_tech(tid);
                if sc >= 160.0 {
                    if best_pre_tech.as_ref().map_or(true, |(b, _)| sc > *b) {
                        best_pre_tech = Some((sc, leg.id.clone()));
                    }
                }
            }
        }
        if let Some((_, id)) = best_pre_tech {
            return id;
        }
    }

    // 4. VILLAGE & ENEMY CITY CAPTURES (kind == 8)
    if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 8) {
        return leg.id.clone();
    }

    // 5. COMBAT: DIRECT ATTACKS & DASH MOVE-ATTACKS
    let mut best_attack: Option<(f32, String)> = None;
    for leg in &obs.legal {
        if leg.cmd.kind == 2 {
            let src = leg.cmd.source as usize;
            let dst = leg.cmd.target as usize;
            if let Some(ui) = obs.unit_at[src] {
                let att = &obs.units[ui];
                let ev = evaluate_attack_from(&obs, mem, att, src, dst, &info);
                let order_bias = if ev.retal == 0.0 && !ev.kills {
                    45.0
                } else if ev.persist_chain {
                    90.0
                } else {
                    0.0
                };
                let total_sc = ev.score + order_bias;
                if total_sc > 25.0 {
                    if best_attack.as_ref().map_or(true, |(b, _)| total_sc > *b) {
                        best_attack = Some((total_sc, leg.id.clone()));
                    }
                }
            }
        }
    }

    let mut best_dash_move: Option<(f32, String)> = None;
    for leg in &obs.legal {
        if leg.cmd.kind == 1 {
            let src = leg.cmd.source as usize;
            let dst = leg.cmd.target as usize;
            if let Some(ui) = obs.unit_at[src] {
                let u = &obs.units[ui];
                let us = unit_stats(u.kind);
                let dt = &obs.tiles[dst];
                let embark = dt.structure == 8 && !us.float;
                let disembark = us.carry && dt.terrain > 2 && !embark;
                // Do not vacate our own city for a Dash move if an enemy unit could step onto our city!
                let vacates_threatened_city = obs.tiles[src].structure == 1
                    && obs.tiles[src].owner == obs.player
                    && obs.units.iter().any(|eu| eu.owner == obs.enemy && cheb(size, eu.tile, src) <= 2);
                if !u.moved && !u.attacked && us.dash && !embark && !disembark && us.attack > 0.0 && !vacates_threatened_city {
                    for eu in &obs.units {
                        if eu.owner == obs.enemy && cheb(size, dst, eu.tile) <= us.range {
                            let ev = evaluate_attack_from(&obs, mem, u, dst, eu.tile, &info);
                            let sc = ev.score - 15.0;
                            if (ev.kills || info.besieged_my_cities.contains(&eu.tile) || ev.retal == 0.0) && sc > 110.0 {
                                if best_dash_move.as_ref().map_or(true, |(b, _)| sc > *b) {
                                    best_dash_move = Some((sc, leg.id.clone()));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((att_sc, att_id)) = &best_attack {
        if let Some((dash_sc, dash_id)) = &best_dash_move {
            if *dash_sc > *att_sc + 120.0 {
                return dash_id.clone();
            }
        }
        if *att_sc >= 75.0 {
            return att_id.clone();
        }
    } else if let Some((dash_sc, dash_id)) = &best_dash_move {
        if *dash_sc >= 130.0 {
            return dash_id.clone();
        }
    }

    // 6. ESCAPE UNITS STEPPING ONTO VILLAGES/CITIES
    for leg in &obs.legal {
        if leg.cmd.kind == 1 {
            let src = leg.cmd.source as usize;
            let dst = leg.cmd.target as usize;
            if let Some(ui) = obs.unit_at[src] {
                let u = &obs.units[ui];
                if u.attacked && !u.moved && obs.tiles[dst].structure == 1 && obs.tiles[dst].owner != obs.player {
                    return leg.id.clone();
                }
            }
        }
    }

    // 7. NAVAL INFRASTRUCTURE PRIORITY
    if info.needs_navy && !info.has_port {
        let mut best_port: Option<(i32, String)> = None;
        for leg in &obs.legal {
            if leg.cmd.kind == 6 && leg.cmd.parameter == 8 {
                let dst = leg.cmd.target as usize;
                let adj_water = neighbors(size, dst)
                    .into_iter()
                    .filter(|&nb| !obs.tiles[nb].known || obs.tiles[nb].terrain == 1 || obs.tiles[nb].terrain == 2)
                    .count() as i32;
                let dist_enemy = cheb(size, dst, info.enemy_target_center);
                let p_score = adj_water * 20 - dist_enemy;
                if best_port.as_ref().map_or(true, |(b, _)| p_score > *b) {
                    best_port = Some((p_score, leg.id.clone()));
                }
            }
        }
        if let Some((_, id)) = best_port {
            return id;
        }
        if !obs.tech[10] && (info.urgent_navy || info.has_shallow_water_in_territory) {
            if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 7 && l.cmd.parameter == 10) {
                return leg.id.clone();
            }
        }
    }

    if info.has_port && !obs.tech[13] && (info.ocean_count > 0 || info.needs_navy) {
        if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 7 && l.cmd.parameter == 13) {
            return leg.id.clone();
        }
    }

    // 8. EXACT CITY LEVEL-UP & HIGH-ROI ECONOMY PLANNER
    let econ_actions = gather_econ_actions(&obs, &info);
    for ea in &econ_actions {
        if ea.is_monument {
            let dst = obs.legal[ea.action_idx].cmd.target as usize;
            let blocks_needed_port = info.needs_navy && !info.has_port && obs.tiles[dst].terrain == 1 && owned_shallow <= 1;
            if !blocks_needed_port {
                return obs.legal[ea.action_idx].id.clone();
            }
        }
    }
    for ea in &econ_actions {
        if ea.stars_per_turn_gain >= 2 {
            return obs.legal[ea.action_idx].id.clone();
        }
    }

    let reserve_for_urgent_port = if info.urgent_navy && !info.has_port && obs.tech[10] && info.has_shallow_water_in_territory && obs.stars < 7 {
        7
    } else {
        0
    };

    if reserve_for_urgent_port == 0 {
        let mut best_city_plan: Option<(f32, usize)> = None;
        for &ct in &info.my_cities {
            let ci = match obs.city_at[ct] {
                Some(i) => i,
                None => continue,
            };
            let city = &obs.cities[ci];
            let needed = (city.level + 1) - city.progress;
            if needed <= 0 {
                continue;
            }
            let mut c_acts: Vec<&EconAction> = econ_actions
                .iter()
                .filter(|ea| ea.city_tile == ct && ea.pop_gain > 0)
                .collect();
            c_acts.sort_by(|a, b| {
                let cpa = (a.cost * 10) / a.pop_gain.max(1);
                let cpb = (b.cost * 10) / b.pop_gain.max(1);
                cpa.cmp(&cpb).then_with(|| b.pop_gain.cmp(&a.pop_gain))
            });
            let mut acc_pop = 0;
            let mut acc_cost = 0;
            let mut first_idx = None;
            for ea in &c_acts {
                if acc_pop >= needed {
                    break;
                }
                acc_pop += ea.pop_gain;
                acc_cost += ea.cost;
                if first_idx.is_none() {
                    first_idx = Some(ea.action_idx);
                }
            }
            if acc_pop >= needed && acc_cost <= obs.stars {
                if city.level >= 4 {
                    if let Some(ui) = obs.unit_at[ct] {
                        let u = &obs.units[ui];
                        if u.owner == obs.player && !u.moved {
                            if obs.legal.iter().any(|l| l.cmd.kind == 1 && l.cmd.source as usize == ct) {
                                continue;
                            }
                        }
                    }
                }
                let reward_bonus = match city.level + 1 {
                    2 => 190.0,
                    3 => 280.0,
                    4 => 230.0,
                    _ => 380.0,
                };
                let plan_score = reward_bonus - (acc_cost as f32) * 12.0;
                if best_city_plan.as_ref().map_or(true, |(b, _)| plan_score > *b) {
                    if let Some(fidx) = first_idx {
                        best_city_plan = Some((plan_score, fidx));
                    }
                }
            }
        }
        if let Some((_, act_idx)) = best_city_plan {
            return obs.legal[act_idx].id.clone();
        }
    }

    // 9. UPGRADE RAFTS IN FRIENDLY TERRITORY BEFORE MOVING THEM!
    {
        let mut best_upgrade: Option<(f32, String)> = None;
        for leg in &obs.legal {
            if leg.cmd.kind == 4 {
                let src = leg.cmd.source as usize;
                let param = leg.cmd.parameter;
                if matches!(param, 41 | 42 | 44) {
                    let u_sc = match param {
                        42 => {
                            let my_scouts = obs.units.iter().filter(|u| u.owner == obs.player && u.kind == 42).count();
                            if my_scouts == 0 { 420.0 } else { 340.0 }
                        }
                        44 => 360.0,
                        41 => {
                            let enemy_near = obs.units.iter().any(|u| u.owner == obs.enemy && cheb(size, u.tile, src) <= 5);
                            if enemy_near || obs.stars >= 18 { 390.0 } else { 250.0 }
                        }
                        _ => 0.0,
                    };
                    if best_upgrade.as_ref().map_or(true, |(b, _)| u_sc > *b) {
                        best_upgrade = Some((u_sc, leg.id.clone()));
                    }
                }
            }
        }
        if let Some((_, id)) = best_upgrade {
            return id;
        }
    }

    // 10. COORDINATED UNIT MOVEMENT & EXPLORATION (kind == 1)
    {
        let unit_targets = assign_unit_targets(&obs, &info);
        let mut best_move: Option<(f32, String, usize)> = None;

        for leg in &obs.legal {
            if leg.cmd.kind != 1 {
                continue;
            }
            let src = leg.cmd.source as usize;
            let dst = leg.cmd.target as usize;
            let u = match obs.unit_at[src] {
                Some(ui) => &obs.units[ui],
                None => continue,
            };
            let us = unit_stats(u.kind);
            let dt = &obs.tiles[dst];

            // Don't move a unit OFF an uncaptured village/enemy city/ruin/starfish!
            let st_src = obs.tiles[src].structure;
            let standing_on_capturable = (st_src == 1 && obs.tiles[src].owner != obs.player)
                || st_src == 2
                || (obs.tiles[src].resource == 8 && obs.tech[14] && st_src == 0);
            if standing_on_capturable {
                continue;
            }

            // Don't move our last defender off our own city if an enemy unit is adjacent or at distance 2 and could capture it!
            if obs.tiles[src].structure == 1 && obs.tiles[src].owner == obs.player {
                let enemy_can_reach_city = obs.units.iter().any(|eu| {
                    eu.owner == obs.enemy && cheb(size, eu.tile, src) <= unit_stats(eu.kind).movement.max(1)
                });
                if enemy_can_reach_city && !(dt.structure == 1 && dt.owner == obs.enemy) {
                    continue;
                }
            }

            let assigned_targ = unit_targets.get(&src).copied().unwrap_or(info.enemy_target_center);
            let dist_map = bfs_dist_from_target(&obs, assigned_targ, us.float, info.has_port || us.float);

            let mut m_score = 0.0f32;

            // Immediate high-value destinations:
            if dt.structure == 1 && dt.owner != obs.player {
                m_score += if dt.owner == obs.enemy { 800.0 } else { 680.0 };
            } else if dt.structure == 2 {
                m_score += 600.0;
            } else if dt.resource == 8 && obs.tech[14] && dt.structure == 0 {
                m_score += 520.0;
            }

            // Embarking onto our Port (structure == 8):
            if dt.structure == 8 && dt.owner == obs.player && !us.float {
                let can_upgrade_immediately = (obs.tech[13] || obs.tech[44]) && obs.stars >= 5;
                if info.needs_navy || info.urgent_navy || assigned_targ == dst {
                    m_score += if can_upgrade_immediately { 450.0 } else if obs.tech[13] { 330.0 } else { 200.0 };
                    if u.kind == 12 {
                        m_score += 200.0;
                    } else if u.health >= 15.0 {
                        m_score += 60.0;
                    }
                } else {
                    m_score -= 150.0;
                }
            }

            if us.float {
                let disembarking = us.carry && dt.terrain >= 3 && dt.structure != 8;
                if disembarking {
                    let near_target = info.enemy_cities.iter().chain(info.neutral_villages.iter()).any(|&t| cheb(size, dst, t) <= 2);
                    let on_my_island = info.my_cities.iter().any(|&mc| cheb(size, dst, mc) <= 2);
                    if dt.structure == 1 && dt.owner != obs.player {
                        m_score += 700.0;
                    } else if near_target {
                        m_score += 380.0;
                    } else if !on_my_island {
                        m_score += 240.0;
                    } else {
                        m_score -= 400.0;
                    }
                } else {
                    let d_before = dist_map[src];
                    let d_after = dist_map[dst];
                    if d_before < 500_000 && d_after < 500_000 {
                        m_score += ((d_before - d_after) as f32) * 35.0;
                    } else {
                        let c0 = cheb(size, src, assigned_targ);
                        let c1 = cheb(size, dst, assigned_targ);
                        m_score += ((c0 - c1) as f32) * 25.0;
                    }

                    let vis_rad = if us.scout { 2 } else { 1 };
                    let mut newly_seen = 0;
                    for ti in 0..obs.n {
                        if cheb(size, dst, ti) <= vis_rad && !obs.tiles[ti].known {
                            newly_seen += 1;
                        }
                    }
                    m_score += (newly_seen as f32) * 24.0;
                }
            } else {
                // Land unit movement toward its uniquely assigned target!
                let d_before = dist_map[src];
                let d_after = dist_map[dst];
                if d_before < 500_000 && d_after < 500_000 {
                    m_score += ((d_before - d_after) as f32) * 35.0;
                } else {
                    let c0 = cheb(size, src, assigned_targ);
                    let c1 = cheb(size, dst, assigned_targ);
                    m_score += ((c0 - c1) as f32) * 28.0;
                }

                // Tie-break by Chebyshev distance to assigned_targ so diagonal moves that make more geometric progress win!
                let c0 = cheb(size, src, assigned_targ);
                let c1 = cheb(size, dst, assigned_targ);
                m_score += ((c0 - c1) as f32) * 12.0;

                // Vision gain (preferring interior unknown tiles over corner edges early game!)
                let vis_rad = if dt.terrain == 4 || us.scout { 2 } else { 1 };
                let mut newly_seen_score = 0.0f32;
                for ti in 0..obs.n {
                    if cheb(size, dst, ti) <= vis_rad && !obs.tiles[ti].known {
                        let (tx, ty) = ((ti % size) as i32, (ti / size) as i32);
                        let s = size as i32;
                        let edge = tx.min(s - 1 - tx).min(ty.min(s - 1 - ty));
                        newly_seen_score += if edge >= 2 { 22.0 } else if edge == 1 { 14.0 } else { 6.0 };
                    }
                }
                m_score += newly_seen_score;

                // Stepping off our own city tile so we can summon a new unit on that city tile!
                if obs.tiles[src].structure == 1 && obs.tiles[src].owner == obs.player {
                    m_score += 65.0;
                }

                // Escape unit (already attacked) retreating from danger:
                if u.attacked {
                    let danger_src = obs.units.iter().any(|eu| eu.owner == obs.enemy && cheb(size, eu.tile, src) <= unit_stats(eu.kind).range.max(1));
                    let danger_dst = obs.units.iter().any(|eu| eu.owner == obs.enemy && cheb(size, eu.tile, dst) <= unit_stats(eu.kind).range.max(1));
                    if danger_src && !danger_dst {
                        m_score += 120.0;
                    } else if danger_dst {
                        m_score -= 90.0;
                    }
                } else {
                    // Unattacked unit: slight bonus for ending on defensive terrain if near enemy
                    let enemy_near_dst = obs.units.iter().any(|eu| eu.owner == obs.enemy && cheb(size, eu.tile, dst) <= 2);
                    if enemy_near_dst && ((dt.terrain == 5 && obs.tech[18]) || (dt.terrain == 4 && obs.tech[20])) {
                        m_score += 20.0;
                    }
                }
            }

            // Only require m_score > 0.0 for normal units (or > 30.0 for Escape units that already attacked)
            let threshold = if u.attacked { 25.0 } else { 0.5 };
            if m_score > threshold {
                if best_move.as_ref().map_or(true, |(b, _, _)| m_score > *b) {
                    best_move = Some((m_score, leg.id.clone(), dst));
                }
            }
        }

        if let Some((_, id, dst)) = best_move {
            mem.visited_targets.insert(dst, obs.turn);
            return id;
        }
    }

    // 11. REMAINING COMBAT ATTACKS
    if let Some((att_sc, att_id)) = best_attack {
        if att_sc > 0.0 {
            return att_id;
        }
    }

    // 12. UNIT SUMMONING WHEN ARMY IS SMALL OR THREATENED (kind == 4 on cities)
    // Summon units BEFORE spending stars on non-urgent tech when our cities are empty and we need expansion/defense!
    let saving_for_naval_tech = (info.urgent_navy && !obs.tech[10] && obs.stars < research_cost_of(&obs, 10, num_my_cities))
        || (info.has_port && !obs.tech[13] && obs.stars < research_cost_of(&obs, 13, num_my_cities));

    let summon_candidate = |obs: &Obs, info: &MapInfo| -> Option<(f32, String)> {
        let mut best_summon: Option<(f32, String)> = None;
        for leg in &obs.legal {
            if leg.cmd.kind == 4 {
                let src = leg.cmd.source as usize;
                let kind = leg.cmd.parameter;
                if matches!(kind, 41 | 42 | 44) {
                    continue;
                }
                let dist_front = info
                    .enemy_cities
                    .iter()
                    .map(|&ec| cheb(size, src, ec))
                    .min()
                    .unwrap_or_else(|| cheb(size, src, info.enemy_target_center));
                let enemy_near_city = obs.units.iter().any(|eu| eu.owner == obs.enemy && cheb(size, eu.tile, src) <= 4);
                let city_has_port = info.city_territory.get(&src).map_or(false, |tiles| {
                    tiles.iter().any(|&ti| obs.tiles[ti].structure == 8 && obs.tiles[ti].owner == obs.player)
                });

                let waiting_land_units = obs.units.iter().filter(|u| u.owner == obs.player && !unit_stats(u.kind).float).count();
                if info.urgent_navy && !enemy_near_city && waiting_land_units >= 3 && !city_has_port {
                    continue;
                }

                let mut s_score = 120.0 - (dist_front as f32) * 2.5;
                if enemy_near_city {
                    s_score += 150.0;
                }
                if city_has_port && info.needs_navy {
                    s_score += 55.0;
                }

                match kind {
                    4 => s_score += if info.urgent_navy { 40.0 } else { 200.0 },
                    11 => s_score += 175.0,
                    8 => {
                        let my_cats = obs.units.iter().filter(|u| u.owner == obs.player && u.kind == 8).count();
                        if !info.urgent_navy && my_cats * 4 <= my_unit_count && !enemy_near_city {
                            s_score += 155.0;
                        } else {
                            s_score -= 60.0;
                        }
                    }
                    5 => {
                        if enemy_near_city || (info.needs_navy && city_has_port) {
                            s_score += 165.0;
                        } else {
                            s_score += 80.0;
                        }
                    }
                    3 => s_score += if !info.urgent_navy { 155.0 } else { 60.0 },
                    9 => s_score += 140.0,
                    38 => {
                        if !info.enemy_cities.is_empty() && !info.urgent_navy && !enemy_near_city {
                            let my_cloaks = obs.units.iter().filter(|u| u.owner == obs.player && u.kind == 38).count();
                            if my_cloaks == 0 {
                                s_score += 160.0;
                            } else {
                                s_score -= 100.0;
                            }
                        } else {
                            s_score -= 100.0;
                        }
                    }
                    2 => s_score += 115.0,
                    _ => {}
                }

                if s_score > 80.0 {
                    if best_summon.as_ref().map_or(true, |(b, _)| s_score > *b) {
                        best_summon = Some((s_score, leg.id.clone()));
                    }
                }
            }
        }
        best_summon
    };

    let any_city_threatened = obs.units.iter().any(|eu| {
        eu.owner == obs.enemy && info.my_cities.iter().any(|&mc| cheb(size, eu.tile, mc) <= 3)
    });
    let need_more_units_now = any_city_threatened || (my_unit_count < (num_my_cities as usize + 2) && !info.urgent_navy);

    if need_more_units_now && reserve_for_urgent_port == 0 && !saving_for_naval_tech {
        // Only let a tech go before summoning if after researching that tech we STILL have >= 3 stars to summon!
        let mut tech_with_summon_leftover: Option<(f32, String)> = None;
        if !any_city_threatened {
            for leg in &obs.legal {
                if leg.cmd.kind == 7 {
                    let tid = leg.cmd.parameter as usize;
                    let cost = research_cost_of(&obs, tid, num_my_cities);
                    if obs.stars - cost >= 3 {
                        let sc = eval_tech(tid);
                        if sc >= 150.0 {
                            if tech_with_summon_leftover.as_ref().map_or(true, |(b, _)| sc > *b) {
                                tech_with_summon_leftover = Some((sc, leg.id.clone()));
                            }
                        }
                    }
                }
            }
        }
        if let Some((_, id)) = tech_with_summon_leftover {
            return id;
        }
        if let Some((_, id)) = summon_candidate(&obs, &info) {
            return id;
        }
    }

    // 13. TECH RESEARCH (kind == 7)
    if reserve_for_urgent_port == 0 {
        let mut best_tech: Option<(f32, String)> = None;
        for leg in &obs.legal {
            if leg.cmd.kind == 7 {
                let tid = leg.cmd.parameter as usize;
                let sc = eval_tech(tid);
                if sc >= 95.0 {
                    if best_tech.as_ref().map_or(true, |(b, _)| sc > *b) {
                        best_tech = Some((sc, leg.id.clone()));
                    }
                }
            }
        }
        if let Some((_, id)) = best_tech {
            return id;
        }
    }

    // 14. HIGH-EFFICIENCY BUILDINGS & INCREMENTAL ECONOMY
    if reserve_for_urgent_port == 0 && !saving_for_naval_tech {
        if let Some((road_act_idx, _)) = find_one_step_city_connection_road(&obs, mem) {
            if obs.stars >= 3 {
                return obs.legal[road_act_idx].id.clone();
            }
        }
        for ea in &econ_actions {
            if ea.pop_gain >= 2 && ea.cost <= 5 && obs.stars >= ea.cost + 2 {
                return obs.legal[ea.action_idx].id.clone();
            }
        }
        let mut best_econ: Option<(f32, usize)> = None;
        for ea in &econ_actions {
            if ea.pop_gain <= 0 {
                continue;
            }
            let ci = match obs.city_at[ea.city_tile] {
                Some(i) => i,
                None => continue,
            };
            let city = &obs.cities[ci];
            let needed = (city.level + 1) - city.progress;
            if obs.stars >= ea.cost + 4 || (needed <= ea.pop_gain + 1 && obs.stars >= ea.cost + 2) {
                let sc = (ea.pop_gain as f32) * 35.0 - (ea.cost as f32) * 8.0 - (needed as f32) * 5.0;
                if best_econ.as_ref().map_or(true, |(b, _)| sc > *b) {
                    best_econ = Some((sc, ea.action_idx));
                }
            }
        }
        if let Some((_, act_idx)) = best_econ {
            return obs.legal[act_idx].id.clone();
        }
    }

    // 15. REMAINING UNIT SUMMONS
    if reserve_for_urgent_port == 0 && !saving_for_naval_tech {
        if let Some((_, id)) = summon_candidate(&obs, &info) {
            return id;
        }
    }

    // 16. HEAL OTHERS & DIPLOMACY
    if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 3 && l.cmd.parameter == 9) {
        return leg.id.clone();
    }
    if obs.stars >= 12 {
        if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 3 && l.cmd.parameter == 20) {
            return leg.id.clone();
        }
    }
    if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 3 && l.cmd.parameter == 19 && l.cmd.target == 0) {
        return leg.id.clone();
    }

    // 17. END TURN (kind == 10)
    if let Some(leg) = obs.legal.iter().find(|l| l.cmd.kind == 10) {
        return leg.id.clone();
    }

    obs.legal
        .iter()
        .find(|l| l.cmd.kind != 11 && !(l.cmd.kind == 3 && l.cmd.parameter == 8))
        .unwrap_or(&obs.legal[0])
        .id
        .clone()
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout());
    let mut mem = BotMemory::new();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            _ => continue,
        };
        let obs: Value = serde_json::from_str(&line).expect("valid observation json");
        let action_id = choose(&obs, &mut mem);
        writeln!(stdout, "{}", json!({"action_id": action_id})).unwrap();
        stdout.flush().unwrap();
    }
}
