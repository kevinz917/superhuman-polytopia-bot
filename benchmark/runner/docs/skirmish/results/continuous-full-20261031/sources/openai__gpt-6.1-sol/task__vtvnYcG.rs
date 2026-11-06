#![allow(dead_code,unused_imports,unused_variables)]
mod geom {

pub const DX: [i32; 8] = [-1, 0, 1, -1, 1, -1, 0, 1];
pub const DY: [i32; 8] = [-1, -1, -1, 0, 0, 1, 1, 1];

#[derive(Clone)]
pub struct Geom {
    pub size: usize,
    pub n: usize,
    /// dist[i*n+j] Chebyshev distance.
    pub dist: Vec<u8>,
    /// nb[i*8+k] neighbour index (clipped into the board like the JAX table) and valid flag.
    pub nb: Vec<u16>,
    pub nb_valid: Vec<bool>,
    /// Per tile: list of in-board neighbour indices (compact form).
    pub adj: Vec<Vec<u16>>,
    pub corners: [usize; 4],
}

impl Geom {
    pub fn new(size: usize) -> Geom {
        let n = size * size;
        let mut dist = vec![0u8; n * n];
        for i in 0..n {
            let (xi, yi) = ((i % size) as i32, (i / size) as i32);
            for j in 0..n {
                let (xj, yj) = ((j % size) as i32, (j / size) as i32);
                dist[i * n + j] = ((xi - xj).abs().max((yi - yj).abs())) as u8;
            }
        }
        let mut nb = vec![0u16; n * 8];
        let mut nb_valid = vec![false; n * 8];
        let mut adj = vec![Vec::with_capacity(8); n];
        let s = size as i32;
        for i in 0..n {
            let (x, y) = ((i % size) as i32, (i / size) as i32);
            for k in 0..8 {
                let (nx, ny) = (x + DX[k], y + DY[k]);
                let valid = nx >= 0 && nx < s && ny >= 0 && ny < s;
                let idx = (ny * s + nx).clamp(0, (n - 1) as i32) as u16;
                nb[i * 8 + k] = idx;
                nb_valid[i * 8 + k] = valid;
                if valid {
                    adj[i].push(idx);
                }
            }
        }
        Geom { size, n, dist, nb, nb_valid, adj, corners: [0, size - 1, n - size, n - 1] }
    }

    #[inline]
    pub fn d(&self, i: usize, j: usize) -> i32 {
        self.dist[i * self.n + j] as i32
    }

    /// Calls f(t) for every tile t with d(center, t) <= radius (row-major order), visiting only that box.
    #[inline]
    pub fn for_ball(&self, center: usize, radius: i32, mut f: impl FnMut(usize)) {
        if radius < 0 {
            return;
        }
        let size = self.size as i32;
        let (cx, cy) = ((center % self.size) as i32, (center / self.size) as i32);
        for y in (cy - radius).max(0)..=(cy + radius).min(size - 1) {
            for x in (cx - radius).max(0)..=(cx + radius).min(size - 1) {
                f((y * size + x) as usize);
            }
        }
    }
}
}
mod state {

/// Most seats a game can have.
pub const MAXP: usize = 4;

#[derive(Clone)]
pub struct Tiles {
    pub terrain: Vec<i32>,
    pub owner: Vec<i32>,
    pub capital: Vec<i32>,
    pub city: Vec<i32>,
    pub road: Vec<bool>,
    pub structure: Vec<i32>,
    pub resource: Vec<i32>,
    pub explored: [Vec<bool>; MAXP],
    pub structure_level: Vec<i32>,
    pub structure_turn: Vec<i32>,
}

#[derive(Clone)]
pub struct Units {
    pub owner: Vec<i32>,
    pub kind: Vec<i32>,
    pub health: Vec<f32>,
    pub veteran: Vec<bool>,
    pub moved: Vec<bool>,
    pub attacked: Vec<bool>,
    pub kills: Vec<i32>,
    pub poison: Vec<bool>,
    pub frozen: Vec<bool>,
    pub boosted: Vec<bool>,
    pub home: Vec<i32>,
    pub created: Vec<i32>,
    pub converted: Vec<bool>,
    pub attacks_performed: Vec<i32>,
    pub prev_x: Vec<i32>,
    pub prev_y: Vec<i32>,
    pub last_attack: Vec<i32>,
    pub passenger: Vec<i32>,
    pub invisible: Vec<bool>,
}

impl Units {
    /// jax.tree.map(lambda x: x.at[i].set(0), units)
    pub fn clear(&mut self, i: usize) {
        self.owner[i] = 0;
        self.kind[i] = 0;
        self.health[i] = 0.0;
        self.veteran[i] = false;
        self.moved[i] = false;
        self.attacked[i] = false;
        self.kills[i] = 0;
        self.poison[i] = false;
        self.frozen[i] = false;
        self.boosted[i] = false;
        self.home[i] = 0;
        self.created[i] = 0;
        self.converted[i] = false;
        self.attacks_performed[i] = 0;
        self.prev_x[i] = 0;
        self.prev_y[i] = 0;
        self.last_attack[i] = 0;
        self.passenger[i] = 0;
        self.invisible[i] = false;
    }
    /// jax.tree.map(lambda x: x.at[t].set(x[s]).at[s].set(0), units)
    pub fn move_unit(&mut self, s: usize, t: usize) {
        macro_rules! mv {
            ($($f:ident),*) => { $( { let v = self.$f[s]; self.$f[t] = v; } )* };
        }
        mv!(owner, kind, health, veteran, moved, attacked, kills, poison, frozen, boosted, home, created, converted,
            attacks_performed, prev_x, prev_y, last_attack, passenger, invisible);
        self.clear(s);
    }
}

#[derive(Clone)]
pub struct Cities {
    pub owner: Vec<i32>,
    pub level: Vec<i32>,
    pub population: Vec<i32>,
    pub progress: Vec<i32>,
    pub production: Vec<i32>,
    pub border: Vec<i32>,
    pub rewards: Vec<i32>,     // [n*9]
    pub territory: Vec<bool>,  // [n*n], row = city tile
    pub connected: Vec<bool>,
    pub order: Vec<i32>,
    pub infiltrated: Vec<i32>,
    pub riot: Vec<bool>,
}

#[derive(Clone)]
pub struct Players {
    pub stars: [i32; MAXP],
    pub score: [i32; MAXP],
    pub tech: [[bool; 128]; MAXP],
    pub tech_turn: [[i32; 128]; MAXP],
    pub known: [[bool; MAXP]; MAXP],
    pub kills: [i32; MAXP],
    pub casualties: [i32; MAXP],
    pub pacifist: [i32; MAXP],
    pub attacked: [bool; MAXP],
    pub built: [[bool; 128]; MAXP],
    pub killed_turn: [i32; MAXP],
    pub resigned_turn: [i32; MAXP],
    pub killer: [i32; MAXP],
    pub peace: [[bool; MAXP]; MAXP],
    pub embassy: [[bool; MAXP]; MAXP],
    pub peace_offer: [[bool; MAXP]; MAXP],
    pub truce_breaker: [[i32; MAXP]; MAXP],
    pub hostile_until: [i32; MAXP],
    pub monuments: [[bool; 128]; MAXP],
    /// Engine-internal (4-player games): 1-based order in which seats left the game (killed or resigned), 0 while
    /// in. Two-player games never set it.
    pub out_order: [i32; MAXP],
}

#[derive(Clone)]
pub struct State {
    pub size: usize,
    pub n: usize,
    /// Seats in this game (2 or 4). Per-player arrays hold MAXP rows; rows >= np stay zero. Two-player games are
    /// bitwise the reference engine's; more seats follow the multi-player rules in docs/four-player-*.md.
    pub np: usize,
    pub tiles: Tiles,
    pub units: Units,
    pub cities: Cities,
    pub players: Players,
    pub current_player: i32,
    pub turn: i32,
    pub version: i32,
    pub state_version: i32,
    pub actions: i32,
    pub terminated: bool,
    pub truncated: bool,
    pub fog: bool,
    pub reference_max_turns: i32,
    pub core_supported: bool,
    pub seed: u32,
    pub initial_seed: u32,
    /// Engine-internal (not a reference State field): structures::remember_monuments has already been applied to
    /// exactly this state, so applying it again is a no-op. Set by core::step; any other way of changing a state
    /// (reset, import) leaves or sets it false.
    pub monuments_settled: bool,
}

impl State {
    pub fn empty(size: usize) -> State {
        State::empty_players(size, 2)
    }

    pub fn empty_players(size: usize, np: usize) -> State {
        assert!((2..=MAXP).contains(&np));
        let n = size * size;
        let z = || vec![0i32; n];
        let b = || vec![false; n];
        State {
            size,
            n,
            np,
            tiles: Tiles {
                terrain: z(), owner: z(), capital: z(), city: vec![-1; n], road: b(), structure: z(), resource: z(),
                explored: [b(), b(), b(), b()], structure_level: z(), structure_turn: z(),
            },
            units: Units {
                owner: z(), kind: z(), health: vec![0.0; n], veteran: b(), moved: b(), attacked: b(), kills: z(),
                poison: b(), frozen: b(), boosted: b(), home: z(), created: z(), converted: b(), attacks_performed: z(),
                prev_x: z(), prev_y: z(), last_attack: z(), passenger: z(), invisible: b(),
            },
            cities: Cities {
                owner: z(), level: z(), population: z(), progress: z(), production: z(), border: z(),
                rewards: vec![0; n * 9], territory: vec![false; n * n], connected: b(), order: z(), infiltrated: z(), riot: b(),
            },
            players: Players {
                stars: [0; MAXP], score: [0; MAXP], tech: [[false; 128]; MAXP], tech_turn: [[0; 128]; MAXP],
                known: [[false; MAXP]; MAXP], kills: [0; MAXP], casualties: [0; MAXP], pacifist: [0; MAXP],
                attacked: [false; MAXP], built: [[false; 128]; MAXP], killed_turn: [0; MAXP], resigned_turn: [0; MAXP],
                killer: [0; MAXP], peace: [[false; MAXP]; MAXP], embassy: [[false; MAXP]; MAXP],
                peace_offer: [[false; MAXP]; MAXP], truce_breaker: [[0; MAXP]; MAXP], hostile_until: [0; MAXP],
                monuments: [[false; 128]; MAXP], out_order: [0; MAXP],
            },
            current_player: 1,
            turn: 0,
            version: 115,
            state_version: 0,
            actions: 0,
            terminated: false,
            truncated: false,
            fog: true,
            reference_max_turns: 45,
            core_supported: true,
            seed: 0,
            initial_seed: 0,
            monuments_settled: false,
        }
    }

    /// Seat index of the player to move (0..np).
    #[inline]
    pub fn p(&self) -> usize {
        (self.current_player - 1) as usize
    }
    /// jnp.clip(owner - 1, 0, np - 1)
    #[inline]
    pub fn pclip(&self, owner: i32) -> usize {
        (owner - 1).clamp(0, self.np as i32 - 1) as usize
    }
    /// JAX gather semantics for `tech[owner - 1]`: a negative index wraps to the last row.
    #[inline]
    pub fn pwrap(&self, owner: i32) -> usize {
        let np = self.np as i32;
        let i = owner - 1;
        (if i < 0 { i + np } else { i }).clamp(0, np - 1) as usize
    }
    /// Seat k (0-based) is still in the game: not killed and not resigned.
    #[inline]
    pub fn alive(&self, k: usize) -> bool {
        self.players.killed_turn[k] == 0 && self.players.resigned_turn[k] == 0
    }
    /// Seats still in the game.
    pub fn alive_count(&self) -> usize {
        (0..self.np).filter(|&k| self.alive(k)).count()
    }
    /// `owner` is another seat's id (not neutral, not the mover's).
    #[inline]
    pub fn foreign(owner: i32, me: i32) -> bool {
        owner > 0 && owner != me
    }
    #[inline]
    pub fn territory(&self, city: usize, tile: usize) -> bool {
        self.cities.territory[city * self.n + tile]
    }
    #[inline]
    pub fn reward(&self, city: usize, k: usize) -> i32 {
        self.cities.rewards[city * 9 + k]
    }
    #[inline]
    pub fn reward_count(&self, city: usize) -> i32 {
        self.cities.rewards[city * 9..city * 9 + 9].iter().sum()
    }
    pub fn pending_reward(&self) -> bool {
        let cp = self.current_player;
        (0..self.n).any(|i| self.cities.owner[i] == cp && self.reward_count(i) < self.cities.level[i] - 1)
    }
}
}
mod hash {

pub fn hash_words(seed: u32, words: &[u32]) -> u32 {
    let mut h = seed.wrapping_add(374761393).wrapping_add(4 * words.len() as u32);
    for &w in words {
        h = h.wrapping_add(w.wrapping_mul(3266489917));
        h = h.rotate_left(17).wrapping_mul(668265263);
    }
    h = (h ^ (h >> 15)).wrapping_mul(2246822519);
    h = (h ^ (h >> 13)).wrapping_mul(3266489917);
    h ^ (h >> 16)
}

#[inline]
pub fn hash_word(seed: u32, word: u32) -> u32 {
    hash_words(seed, &[word])
}

const MAX: i32 = 2147483647;

pub struct DotNetRandom {
    values: [i32; 56],
    current: usize,
    lag: usize,
}

impl DotNetRandom {
    pub fn new(seed: i32) -> DotNetRandom {
        let absolute = if seed == i32::MIN { MAX } else { seed.abs() };
        let first = 161803398i32.wrapping_sub(absolute);
        let mut values = [0i32; 56];
        values[55] = first;
        let (mut previous, mut next_value) = (first, 1i32);
        for i in 1..55usize {
            let at = (21 * i) % 55;
            values[at] = next_value;
            let delta = previous.wrapping_sub(next_value);
            previous = next_value;
            next_value = if delta < 0 { delta.wrapping_add(MAX) } else { delta };
        }
        for _ in 0..4 {
            for i in 1..56usize {
                let delta = values[i].wrapping_sub(values[1 + (i + 30) % 55]);
                values[i] = if delta < 0 { delta.wrapping_add(MAX) } else { delta };
            }
        }
        DotNetRandom { values, current: 0, lag: 21 }
    }

    pub fn next_int(&mut self) -> i32 {
        let i = if self.current == 55 { 1 } else { self.current + 1 };
        let j = if self.lag == 55 { 1 } else { self.lag + 1 };
        let mut value = self.values[i].wrapping_sub(self.values[j]);
        if value == MAX {
            value -= 1;
        }
        if value < 0 {
            value = value.wrapping_add(MAX);
        }
        self.values[i] = value;
        self.current = i;
        self.lag = j;
        value
    }

    /// Exact Next(0, upper) with the reference's 32-bit limb arithmetic.
    pub fn bounded(&mut self, upper: i32) -> i32 {
        let a = self.next_int() as u32;
        let b = upper as u32;
        let lower = (a & 65535).wrapping_mul(b);
        let middle = (a >> 16).wrapping_mul(b).wrapping_add(lower >> 16);
        let high = middle >> 16;
        let low = (middle << 16) | (lower & 65535);
        let mut quotient = (high << 1) | (low >> 31);
        let remainder = (low & 2147483647).wrapping_add(quotient);
        quotient = quotient.wrapping_add((remainder >= 2147483647) as u32);
        quotient as i32
    }

    /// Float32 cast of NextDouble, including the upward-shifted halfway cases.
    pub fn unit_float(&mut self) -> f32 {
        let value = self.next_int();
        let bits = value as u32;
        let shift = (8 - bits.leading_zeros() as i32).max(0) as u32;
        let quantum = 1u32 << shift;
        let tie = shift > 0 && (bits & (quantum - 1)) == (quantum >> 1) && ((bits / quantum) & 1) == 0;
        let result = (value as f32) * (1.0f32 / 2147483648.0f32);
        if tie {
            f32::from_bits(result.to_bits() + 1)
        } else {
            result
        }
    }
}
}
mod cov {

use std::cell::RefCell;

pub const NAMES: [&str; 40] = [
    "research", "recover", "move", "end_turn", "attack", "convert", "infiltrate", "summon", "upgrade", "harvest",
    "forest", "promote", "disband", "reward", "build", "destroy", "capture_city", "capture_enemy_city", "ruin_stars",
    "ruin_tech", "ruin_population", "ruin_explore", "ruin_unit", "starfish", "heal_others", "diplomacy", "resign",
    "splash", "stomp", "push", "elimination", "truncated_round_cap", "truncated_max_actions", "move_hidden_encounter",
    "retaliation_kill", "advance_after_kill", "embark_disembark", "area_damage_kill", "reward_giant", "reward_explore",
];

thread_local! {
    static COV: RefCell<[u64; 40]> = const { RefCell::new([0; 40]) };
}

#[inline]
pub fn hit(i: usize) {
    COV.with(|c| c.borrow_mut()[i] += 1);
}

pub fn snapshot() -> [u64; 40] {
    COV.with(|c| *c.borrow())
}

pub fn reset() {
    COV.with(|c| *c.borrow_mut() = [0; 40]);
}
}
mod prof {

use std::cell::{Cell, RefCell};
use std::time::Instant;

pub const KINDS: [&str; 18] = [
    "research", "recover", "move", "end_turn", "attack", "summon", "harvest", "forest", "promote", "disband", "reward",
    "build", "destroy", "capture", "heal", "diplomacy", "resign", "unsupported",
];
/// Phase ids: 0..18 step validation per kind, 18..36 step execution per kind, then the fixed phases below.
pub const VALIDATE: usize = 0;
pub const EXECUTE: usize = 18;
pub const STEP_PRE: usize = 36;
pub const STEP_POST: usize = 37;
pub const TICK_STEP: usize = 38;
pub const TICK_RESET: usize = 39;
pub const TICK_MASKS: usize = 40;
pub const TICK_OBSERVE: usize = 41;
pub const TICK_OBSERVE_OPP: usize = 42;
pub const TICK_FINAL: usize = 43;
pub const TICK_ALLOC: usize = 44;
pub const TICK_OUTPUT: usize = 45;
pub const N: usize = 46;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static TOTAL: RefCell<[(u64, u64); N]> = const { RefCell::new([(0, 0); N]) };
}

pub fn enable(on: bool) {
    ON.with(|c| c.set(on));
    TOTAL.with(|t| *t.borrow_mut() = [(0, 0); N]);
}

/// Run `f`, adding its wall time to phase `id` when profiling is on.
#[inline]
pub fn time<T>(id: usize, f: impl FnOnce() -> T) -> T {
    if !ON.with(|c| c.get()) {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    let ns = t0.elapsed().as_nanos() as u64;
    TOTAL.with(|t| {
        let mut t = t.borrow_mut();
        t[id].0 += 1;
        t[id].1 += ns;
    });
    out
}

pub fn name(id: usize) -> String {
    match id {
        i if i < EXECUTE => format!("validate.{}", KINDS[i]),
        i if i < STEP_PRE => format!("execute.{}", KINDS[i - EXECUTE]),
        STEP_PRE => "step.pre".into(),
        STEP_POST => "step.post".into(),
        TICK_STEP => "tick.step".into(),
        TICK_RESET => "tick.reset".into(),
        TICK_MASKS => "tick.masks".into(),
        TICK_OBSERVE => "tick.observe".into(),
        TICK_OBSERVE_OPP => "tick.observe_opp".into(),
        TICK_FINAL => "tick.final_view".into(),
        TICK_ALLOC => "tick.alloc".into(),
        _ => "tick.output".into(),
    }
}

/// (name, calls, total ms) for every phase that ran.
pub fn snapshot() -> Vec<(String, u64, f64)> {
    TOTAL.with(|t| {
        t.borrow().iter().enumerate().filter(|(_, v)| v.0 > 0).map(|(i, v)| (name(i), v.0, v.1 as f64 / 1e6)).collect()
    })
}
}
mod rules {

struct UnitRow { id:usize,name:String,health:f32,attack:f32,defense:f32,movement:i32,range:i32,cost:i32,skills:Vec<String> }
struct TechRow { id:i64,tier:i32,requires:Option<i64>,next:Vec<i64>,structures:Vec<usize> }
struct StructRow { id:usize,cost:Option<i32>,population:Option<i32>,score:Option<i32>,stars:Option<i32>,resource:Option<i32>,terrain:Vec<usize>,adjacent:Vec<usize>,per_city:bool,per_tribe:bool,tribe:Option<i64> }
struct Table {engine_revision:String,units:Vec<UnitRow>,technologies:Vec<TechRow>,structures:Vec<StructRow>}
impl Table {fn parse(v:&serde_json::Value)->Self {
 let arr=|k:&str| v[k].as_array().unwrap();
 let nums=|v:&serde_json::Value|->Vec<usize>{v.as_array().unwrap().iter().map(|x|x.as_u64().unwrap() as usize).collect()};
 Self{engine_revision:v["engine_revision"].as_str().unwrap().to_owned(),
 units:arr("units").iter().map(|u|UnitRow{id:u["id"].as_u64().unwrap() as usize,name:u["name"].as_str().unwrap().to_owned(),health:u["health"].as_f64().unwrap() as f32,attack:u["attack"].as_f64().unwrap() as f32,defense:u["defense"].as_f64().unwrap() as f32,movement:u["movement"].as_i64().unwrap() as i32,range:u["range"].as_i64().unwrap() as i32,cost:u["cost"].as_i64().unwrap() as i32,skills:u["skills"].as_array().unwrap().iter().map(|x|x.as_str().unwrap().to_owned()).collect()}).collect(),
 technologies:arr("technologies").iter().map(|u|TechRow{id:u["id"].as_i64().unwrap(),tier:u["tier"].as_i64().unwrap() as i32,requires:u["requires"].as_i64(),next:u["next"].as_array().unwrap().iter().map(|x|x.as_i64().unwrap()).collect(),structures:nums(&u["structures"])}).collect(),
 structures:arr("structures").iter().map(|u|StructRow{id:u["id"].as_u64().unwrap() as usize,cost:u["cost"].as_i64().map(|x|x as i32),population:u["population"].as_i64().map(|x|x as i32),score:u["score"].as_i64().map(|x|x as i32),stars:u["stars"].as_i64().map(|x|x as i32),resource:u["resource"].as_i64().map(|x|x as i32),terrain:nums(&u["terrain"]),adjacent:nums(&u["adjacent"]),per_city:u["per_city"].as_bool().unwrap(),per_tribe:u["per_tribe"].as_bool().unwrap(),tribe:u["tribe"].as_i64()}).collect()}
}}



pub const FLOAT: usize = 0;
pub const CARRY: usize = 1;
pub const SCOUT: usize = 2;
pub const CREEP: usize = 3;
pub const HIDE: usize = 4;
pub const INDEPENDENT: usize = 5;
pub const SURPRISE: usize = 6;
pub const SPLASH: usize = 7;
pub const STOMP: usize = 8;
pub const INFILTRATE: usize = 9;

/// For a cost table: the kinds whose cost is at most a given number of stars, as a bitset (`le(stars)`).
pub struct CostIndex {
    min: i32,
    table: Vec<u128>,
}

impl CostIndex {
    pub fn new(cost: &[i32; 128]) -> CostIndex {
        let (min, max) = (*cost.iter().min().unwrap(), *cost.iter().max().unwrap());
        let table = (min..=max).map(|s| (0..128).fold(0u128, |b, k| b | ((cost[k] <= s) as u128) << k)).collect();
        CostIndex { min, table }
    }
    #[inline]
    pub fn le(&self, stars: i32) -> u128 {
        if stars < self.min {
            0
        } else {
            self.table[((stars - self.min) as usize).min(self.table.len() - 1)]
        }
    }
}

pub struct StructureRules {
    pub cost: [i32; 128],
    pub population: [i32; 128],
    pub score: [i32; 128],
    pub stars: [i32; 128],
    pub terrain: [[bool; 16]; 128],
    pub adjacent: [[bool; 128]; 128],
    pub any_adjacent: [bool; 128],
    pub resource: [i32; 128],
    pub per_city: [bool; 128],
    pub per_tribe: [bool; 128],
    pub allowed: [bool; 128],
    pub unlock: [[bool; 128]; 128], // [tech, structure]
    /// Derived bitsets: unlock_by[k] bit i = unlock[i][k]; adjacent_rev[j] bit k = adjacent[k][j];
    /// terrain_bits[t] bit k = terrain[k][t]; no_adjacent_bits / per_city_bits bit k = !any_adjacent[k] / per_city[k].
    pub unlock_by: [u128; 128],
    pub adjacent_rev: [u128; 128],
    pub terrain_bits: [u128; 16],
    pub no_adjacent_bits: u128,
    pub per_city_bits: u128,
    /// tech_unlocks[i]: structures tech i unlocks; static_ok: allowed && cost >= 0 && no resource; per_tribe_bits;
    /// cost_le: kinds affordable with a given number of stars.
    pub tech_unlocks: [u128; 128],
    pub static_ok: u128,
    pub per_tribe_bits: u128,
    pub cost_le: CostIndex,
}

pub struct Rules {
    pub unit_health: [f32; 128],
    pub unit_attack: [f32; 128],
    pub unit_defense: [f32; 128],
    pub unit_movement: [i32; 128],
    pub unit_range: [i32; 128],
    pub dash: [bool; 128],
    pub unit_supported: [bool; 128],
    pub tech_tier: [i32; 128],
    pub tech_edges: [[bool; 128]; 128],
    pub tech_exists: [bool; 128],
    pub unit_cost: [i32; 128],
    pub fortify: [bool; 128],
    pub stiff: [bool; 128],
    pub persist: [bool; 128],
    pub escape: [bool; 128],
    pub stat: [bool; 128],
    pub unit_tech: [i32; 128],
    pub structures: StructureRules,
    pub free_tech: [bool; 128],
    pub free_requires: [i32; 128],
    pub skills: [[bool; 10]; 128],
    pub engine_revision: String,
    /// Derived: tech_pred[k] bit i = tech_edges[i][k] (research reachability).
    pub tech_pred: [u128; 128],
    /// Derived: bits128(unit_supported), bits128(tech_exists).
    pub supported_bits: u128,
    pub tech_exists_bits: u128,
    /// Derived: tech_succ[i] bit k = tech_edges[i][k]; units_by_tech[i]: unit kinds tech i unlocks (cost >= 1,
    /// unit_tech < 127); unit_static: the Warrior (kind 2) when summonable without a tech; unit_cost_le.
    pub tech_succ: [u128; 128],
    pub units_by_tech: [u128; 128],
    pub unit_static: u128,
    pub unit_cost_le: CostIndex,
}

/// Bit k set for every true entry of a 128-wide flag row.
#[inline]
pub fn bits128(row: &[bool; 128]) -> u128 {
    let mut b = 0u128;
    for (k, &v) in row.iter().enumerate() {
        b |= (v as u128) << k;
    }
    b
}

impl Rules {
    pub fn from_json(text: &str) -> Result<Rules, String> {
        let data: serde_json::Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let data = Table::parse(&data);
        let mut r = Rules {
            unit_health: [0.0; 128],
            unit_attack: [0.0; 128],
            unit_defense: [0.0; 128],
            unit_movement: [0; 128],
            unit_range: [0; 128],
            dash: [false; 128],
            unit_supported: [false; 128],
            tech_tier: [1; 128],
            tech_edges: [[false; 128]; 128],
            tech_exists: [false; 128],
            unit_cost: [0; 128],
            fortify: [false; 128],
            stiff: [false; 128],
            persist: [false; 128],
            escape: [false; 128],
            stat: [false; 128],
            unit_tech: [127; 128],
            structures: StructureRules {
                cost: [-1; 128],
                population: [0; 128],
                score: [0; 128],
                stars: [0; 128],
                terrain: [[false; 16]; 128],
                adjacent: [[false; 128]; 128],
                any_adjacent: [false; 128],
                resource: [0; 128],
                per_city: [false; 128],
                per_tribe: [false; 128],
                allowed: [false; 128],
                unlock: [[false; 128]; 128],
                unlock_by: [0; 128],
                adjacent_rev: [0; 128],
                terrain_bits: [0; 16],
                no_adjacent_bits: 0,
                per_city_bits: 0,
                tech_unlocks: [0; 128],
                static_ok: 0,
                per_tribe_bits: 0,
                cost_le: CostIndex { min: 0, table: vec![0] },
            },
            free_tech: [false; 128],
            free_requires: [-1; 128],
            skills: [[false; 10]; 128],
            engine_revision: data.engine_revision.clone(),
            tech_pred: [0; 128],
            supported_bits: 0,
            tech_exists_bits: 0,
            tech_succ: [0; 128],
            units_by_tech: [0; 128],
            unit_static: 0,
            unit_cost_le: CostIndex { min: 0, table: vec![0] },
        };
        let core_names = [
            "Raft", "Rammership", "Scoutship", "Bomber", "Juggernaut", "MindBender", "Warrior", "Rider", "Knight",
            "Defender", "Catapult", "Archer", "Swordsman", "Giant", "Cloak", "Dagger", "Dinghy", "Pirate",
        ];
        let skill_names = ["Float", "Carry", "Scout", "Creep", "Hide", "Independent", "Surprise", "Splash", "Stomp", "Infiltrate"];
        for u in &data.units {
            let i = u.id;
            if i >= 128 {
                return Err("unit id out of range".into());
            }
            r.unit_health[i] = u.health;
            r.unit_attack[i] = u.attack;
            r.unit_defense[i] = u.defense;
            r.unit_movement[i] = u.movement;
            r.unit_range[i] = u.range;
            r.dash[i] = u.skills.iter().any(|s| s == "Dash");
            r.unit_supported[i] = core_names.contains(&u.name.as_str());
            for (k, name) in skill_names.iter().enumerate() {
                r.skills[i][k] = u.skills.iter().any(|s| s == name);
            }
            r.unit_cost[i] = u.cost;
            r.fortify[i] = u.skills.iter().any(|s| s == "Fortify");
            r.stiff[i] = u.skills.iter().any(|s| s == "Stiff");
            r.persist[i] = u.skills.iter().any(|s| s == "Persist");
            r.escape[i] = u.skills.iter().any(|s| s == "Escape");
            r.stat[i] = u.skills.iter().any(|s| s == "Static");
        }
        for t in &data.technologies {
            if t.id < 0 {
                continue;
            }
            let i = t.id as usize;
            if i >= 128 {
                return Err("tech id out of range".into());
            }
            r.tech_tier[i] = t.tier;
            r.tech_exists[i] = true;
            let mut others = t.next.clone();
            if let Some(req) = t.requires {
                others.push(req);
            }
            for o in others {
                if o >= 0 && (o as usize) < 128 {
                    r.tech_edges[i][o as usize] = true;
                }
            }
        }
        for i in 25..128 {
            r.tech_exists[i] = false;
        }
        r.tech_exists[38] = true;
        r.tech_exists[44] = true;
        for (unit, tech) in [(2, 0), (3, 1), (4, 3), (5, 7), (8, 17), (9, 18), (10, 22), (11, 24), (38, 38), (41, 14), (42, 13), (44, 44)] {
            r.unit_tech[unit] = tech;
        }
        for st in &data.structures {
            let i = st.id;
            if i >= 128 {
                return Err("structure id out of range".into());
            }
            let s = &mut r.structures;
            s.cost[i] = st.cost.unwrap_or(-1);
            s.population[i] = st.population.unwrap_or(0);
            s.score[i] = st.score.unwrap_or(0);
            s.stars[i] = st.stars.unwrap_or(0);
            s.resource[i] = st.resource.unwrap_or(0);
            for &t in &st.terrain {
                if t < 16 {
                    s.terrain[i][t] = true;
                }
            }
            for &a in &st.adjacent {
                if a < 128 {
                    s.adjacent[i][a] = true;
                    s.any_adjacent[i] = true;
                }
            }
            s.per_city[i] = st.per_city;
            s.per_tribe[i] = st.per_tribe;
            s.allowed[i] = st.tribe.is_none() || st.tribe == Some(4);
        }
        for t in &data.technologies {
            if t.id >= 0 {
                for &st in &t.structures {
                    if st < 128 {
                        r.structures.unlock[t.id as usize][st] = true;
                    }
                }
            }
        }
        for t in &data.technologies {
            let i = t.id;
            if (1..=24).contains(&i) || i == 38 || i == 44 || i == 48 || i == 121 {
                r.free_tech[i as usize] = true;
                r.free_requires[i as usize] = t.requires.map(|v| v as i32).unwrap_or(-1);
            }
        }
        for i in 0..128 {
            r.free_tech[i] &= r.tech_exists[i];
        }
        let rs = &mut r.structures;
        for k in 0..128 {
            for i in 0..128 {
                r.tech_pred[k] |= (r.tech_edges[i][k] as u128) << i;
                rs.unlock_by[k] |= (rs.unlock[i][k] as u128) << i;
                rs.adjacent_rev[i] |= (rs.adjacent[k][i] as u128) << k;
            }
            for t in 0..16 {
                rs.terrain_bits[t] |= (rs.terrain[k][t] as u128) << k;
            }
            rs.no_adjacent_bits |= (!rs.any_adjacent[k] as u128) << k;
            rs.per_city_bits |= (rs.per_city[k] as u128) << k;
        }
        r.supported_bits = bits128(&r.unit_supported);
        r.tech_exists_bits = bits128(&r.tech_exists);
        for i in 0..128 {
            r.tech_succ[i] = bits128(&r.tech_edges[i]);
            r.structures.tech_unlocks[i] = bits128(&r.structures.unlock[i]);
        }
        let rs = &mut r.structures;
        for k in 0..128 {
            rs.static_ok |= ((rs.allowed[k] && rs.cost[k] >= 0 && rs.resource[k] == 0) as u128) << k;
            rs.per_tribe_bits |= (rs.per_tribe[k] as u128) << k;
            let summonable = r.unit_cost[k] >= 1 && r.unit_tech[k] < 127;
            if summonable {
                r.units_by_tech[r.unit_tech[k] as usize] |= 1u128 << k;
                if k == 2 {
                    r.unit_static |= 1u128 << k;
                }
            }
        }
        rs.cost_le = CostIndex::new(&rs.cost);
        r.unit_cost_le = CostIndex::new(&r.unit_cost);
        Ok(r)
    }
}
}
mod economy {

use crate::geom::Geom;
use crate::rules::Rules;
use crate::state::State;

pub fn add_population(s: &mut State, index: usize, amount: i32) {
    let n = s.n;
    let cp = s.current_player;
    let mut v = vec![0i32; n];
    if index < n && s.cities.owner[index] == cp {
        v[index] = amount;
    }
    add_population_vector(s, &v, cp, false);
}

/// economy.add_population_vector: level-ups (and, when !native, level-downs) per city, score to `owner`.
pub fn add_population_vector(s: &mut State, amount: &[i32], owner: i32, native: bool) {
    let n = s.n;
    let mut score: i32 = 0;
    for i in 0..n {
        let active = s.cities.owner[i] == owner && amount[i] != 0;
        let a = if active { amount[i] } else { 0 };
        let old_level = s.cities.level[i];
        let c = &mut s.cities;
        c.progress[i] += a;
        if active {
            while c.progress[i] >= c.level[i] + 1 {
                c.progress[i] -= c.level[i] + 1;
                c.level[i] += 1;
                c.production[i] += 1;
                if s.tiles.structure[i] > 0 {
                    s.tiles.structure_level[i] += 1;
                }
            }
            if !native {
                while c.progress[i] < 0 && c.level[i] > 1 {
                    c.progress[i] += c.level[i];
                    c.level[i] -= 1;
                    c.production[i] = (c.production[i] - 1).max(1);
                    if s.tiles.structure[i] > 0 {
                        s.tiles.structure_level[i] = (s.tiles.structure_level[i] - 1).max(1);
                    }
                }
            }
        }
        c.population[i] += a;
        score += a * 5 + (c.level[i] - old_level) * 50;
    }
    let o = (owner - 1) as usize;
    s.players.score[o] = s.players.score[o].wrapping_add(score);
}

pub fn in_territory(s: &State, out: &mut [bool]) {
    let n = s.n;
    out[..n].fill(false);
    for c in 0..n {
        if s.cities.owner[c] == s.current_player {
            let row = &s.cities.territory[c * n..c * n + n];
            for t in 0..n {
                out[t] |= row[t];
            }
        }
    }
}

pub fn economic_tiles(s: &State, out: &mut [bool]) {
    in_territory(s, out);
    for t in 0..s.n {
        out[t] = out[t] && unit_allows_work(s, t) && s.tiles.owner[t] == s.current_player;
    }
}

/// No unit blocks work on tile `t`: none, the mover's, a hidden one, or (GameLogicData.CanBuild /
/// TileData.CanDestroy) a unit of a player the mover has peace with.
#[inline]
pub fn unit_allows_work(s: &State, t: usize) -> bool {
    let o = s.units.owner[t];
    o == 0 || o == s.current_player || s.units.invisible[t] || s.players.peace[s.p()][s.pclip(o)]
}

pub fn harvest_mask(s: &State, out: &mut [bool]) {
    let mut econ = vec![false; s.n];
    economic_tiles(s, &mut econ);
    harvest_mask_with(s, &econ, out);
}

/// harvest_mask given economic_tiles(s).
pub fn harvest_mask_with(s: &State, econ: &[bool], out: &mut [bool]) {
    let n = s.n;
    let tech = &s.players.tech[s.p()];
    let stars = s.players.stars[s.p()];
    for t in 0..n {
        let r = s.tiles.resource[t];
        let unlocked = (r == 1 && tech[15]) || (r == 3 && tech[10]) || (r == 6 && tech[6]);
        out[t] = econ[t] && s.tiles.structure[t] == 0 && unlocked && stars >= 2;
    }
}

pub fn harvest(s: &mut State, target: usize) {
    let city = s.tiles.city[target].clamp(0, s.n as i32 - 1) as usize;
    s.tiles.resource[target] = 0;
    let p = s.p();
    s.players.stars[p] -= 2;
    add_population(s, city, 1);
}

/// out[t*3 + k]
pub fn forest_masks(s: &State, out: &mut [bool]) {
    let mut econ = vec![false; s.n];
    economic_tiles(s, &mut econ);
    forest_masks_with(s, &econ, out);
}

/// forest_masks given economic_tiles(s).
pub fn forest_masks_with(s: &State, econ: &[bool], out: &mut [bool]) {
    let n = s.n;
    let tech = &s.players.tech[s.p()];
    let stars = s.players.stars[s.p()];
    for t in 0..n {
        let free = econ[t] && (s.tiles.structure[t] == 0 || s.tiles.structure[t] == 3);
        let forest = s.tiles.terrain[t] == 5;
        let field = s.tiles.terrain[t] == 3;
        out[t * 3] = free && forest && tech[9] && stars >= 3;
        out[t * 3 + 1] = free && forest && tech[16];
        out[t * 3 + 2] = free && field && tech[19] && stars >= 5;
    }
}

pub fn forest(s: &mut State, target: usize, ability: i32) {
    let terrain = if ability == 3 { 5 } else { 3 };
    // BuildAction: a `creates` terrain clears the resource (Consumed improvements destroy it), so Grow
    // Forest leaves no resource behind
    let resource = if ability == 1 { 2 } else { 0 };
    let stars = if ability == 1 { -3 } else if ability == 2 { 1 } else { -5 };
    s.tiles.terrain[target] = terrain;
    s.tiles.resource[target] = resource;
    let p = s.p();
    s.players.stars[p] += stars;
}

/// economy.discover(s, reveal, player): exploration gain for `player`; lighthouse population to the
/// preferred city of `player` (applied only if that city belongs to the player to move).
pub fn discover(s: &mut State, reveal: &[bool], player: i32) {
    let n = s.n;
    let p = (player - 1) as usize;
    let mut new = 0i32;
    let mut count = 0i32;
    for t in 0..n {
        if reveal[t] && !s.tiles.explored[p][t] {
            new += 1;
            if s.tiles.structure[t] == 47 {
                count += 1;
            }
            s.tiles.explored[p][t] = true;
        }
    }
    s.players.score[p] = s.players.score[p].wrapping_add(new * 5);
    if count == 0 {
        return; // add_population(.., 0) below changes nothing
    }
    let c = &s.cities;
    let any_capital = (0..n).any(|i| c.owner[i] == player && s.tiles.capital[i] == player);
    let any_own = (0..n).any(|i| c.owner[i] == player);
    let mut index = 0usize;
    let mut best = i32::MAX;
    for i in 0..n {
        let preferred = if any_capital { c.owner[i] == player && s.tiles.capital[i] == player } else { c.owner[i] == player };
        if preferred && c.order[i] < best {
            best = c.order[i];
            index = i;
        }
    }
    add_population(s, index, if any_own { count } else { 0 });
}

/// out[t*9 + k]
pub fn reward_mask(s: &State, out: &mut [bool]) {
    let n = s.n;
    for t in 0..n {
        let count = s.reward_count(t);
        let pending = s.cities.owner[t] == s.current_player && count < s.cities.level[t] - 1;
        let slot = count + 1;
        for k in 0..9i32 {
            let allowed = match slot {
                1 => k == 3 || k == 4,
                2 => k == 1 || k == 7,
                3 => k == 5 || k == 8,
                _ => k == 2 || k == 6,
            };
            out[t * 9 + k as usize] = pending && allowed;
        }
    }
}

pub fn claim(s: &mut State, city: usize, mask: &[bool], r: &Rules) {
    let n = s.n;
    let cp = s.current_player;
    let (mut gain, mut loss) = (0i32, [0i32; crate::state::MAXP]);
    for t in 0..n {
        if mask[t] && s.tiles.owner[t] != cp {
            let value = 20 + crate::structures::score_of(s, r, t);
            gain += value;
            if State::foreign(s.tiles.owner[t], cp) {
                loss[(s.tiles.owner[t] - 1) as usize] += value;
            }
        }
    }
    for t in 0..n {
        if mask[t] {
            s.tiles.owner[t] = cp;
            s.tiles.city[t] = city as i32;
            s.cities.territory[city * n + t] = true;
        }
    }
    let p = s.p();
    s.players.score[p] = s.players.score[p].wrapping_add(gain);
    // with two seats exactly the reference's single subtraction from the opponent (also when its loss is 0)
    for q in (0..s.np).filter(|&q| q != p) {
        s.players.score[q] = s.players.score[q].wrapping_sub(loss[q]);
    }
}

pub fn reward(s: &mut State, target: usize, kind: i32, r: &Rules, g: &Geom) {
    let n = s.n;
    s.cities.rewards[target * 9 + kind as usize] += 1;
    match kind {
        2 | 3 => {
            s.cities.production[target] += 1;
            if kind == 2 {
                let p = s.p();
                s.players.score[p] += 250;
            }
        }
        7 => {
            let p = s.p();
            s.players.stars[p] += 5;
        }
        8 => add_population(s, target, 3),
        5 => {
            let mut area = vec![false; n];
            for t in 0..n {
                area[t] = g.d(target, t) <= 2 && t != target;
            }
            s.cities.border[target] += 1;
            let claimed: Vec<bool> = (0..n).map(|t| area[t] && s.tiles.owner[t] == 0).collect();
            claim(s, target, &claimed, r);
            discover(s, &area, s.current_player);
        }
        4 => {
            let reveal = crate::discovery::explorer(s, target, g);
            discover(s, &reveal, s.current_player);
        }
        6 => crate::military::superunit(s, target, r, g),
        _ => {}
    }
}
}
mod structures {

use crate::economy::{add_population, add_population_vector, economic_tiles, in_territory};
use crate::geom::Geom;
use crate::rules::{bits128, Rules};
use crate::state::State;

#[inline]
pub fn temple(kind: i32) -> bool {
    (17..=20).contains(&kind)
}

#[inline]
pub fn score_of(s: &State, r: &Rules, t: usize) -> i32 {
    let k = s.tiles.structure[t];
    if temple(k) { 100 * s.tiles.structure_level[t] } else { r.structures.score[k as usize] }
}

/// Native multiplier levels from adjacent owned compatible improvements.
pub fn production_levels(s: &State, r: &Rules, g: &Geom) -> Vec<i32> {
    let n = s.n;
    let t = &s.tiles;
    let mut levels = vec![0i32; n];
    for i in 0..n {
        let k = t.structure[i] as usize;
        let multiplier = r.structures.any_adjacent[k] && t.structure[i] != 50;
        if multiplier {
            let mut count = 0;
            for &j in &g.adj[i] {
                let j = j as usize;
                if t.owner[i] == t.owner[j] && t.owner[i] > 0 && r.structures.adjacent[k][t.structure[j] as usize] {
                    count += 1;
                }
            }
            levels[i] = count;
        } else {
            levels[i] = t.structure_level[i];
        }
    }
    if !t.structure.contains(&50) {
        return levels;
    }
    let mut out = levels.clone();
    for i in 0..n {
        if t.structure[i] == 50 {
            let mut sum = 0;
            for &j in &g.adj[i] {
                let j = j as usize;
                if t.owner[i] == t.owner[j] && t.owner[i] > 0 && r.structures.adjacent[50][t.structure[j] as usize] {
                    sum += levels[j];
                }
            }
            out[i] = sum.min(8);
        }
    }
    out
}

pub fn population_contribution(s: &State, r: &Rules, g: &Geom) -> Vec<i32> {
    contribution_from(s, r, &production_levels(s, r, g))
}

/// population_contribution given production_levels(s).
fn contribution_from(s: &State, r: &Rules, levels: &[i32]) -> Vec<i32> {
    (0..s.n)
        .map(|i| {
            let k = s.tiles.structure[i] as usize;
            r.structures.population[k] * if r.structures.any_adjacent[k] { levels[i] } else { 1 }
        })
        .collect()
}

/// structures.adjust_building_population(before, after): `before_contribution` is population_contribution of the
/// pre-action state.
pub fn adjust_building_population(before_contribution: &[i32], s: &mut State, r: &Rules, g: &Geom) {
    let n = s.n;
    let levels = production_levels(s, r, g);   // s is unchanged until structure_level is set below
    let after = contribution_from(s, r, &levels);
    let mut amount = vec![0i32; n];
    for tile in 0..n {
        let city = s.tiles.city[tile];
        if city >= 0 && (city as usize) < n {
            let city = city as usize;
            if s.cities.owner[city] == s.tiles.owner[tile] {
                amount[city] += after[tile] - before_contribution[tile];
            }
        }
    }
    s.tiles.structure_level = levels;
    if amount.iter().any(|&a| a != 0) {   // all-zero amounts change nothing
        add_population_vector(s, &amount, 1, true);
        add_population_vector(s, &amount, 2, true);
    }
}

pub fn capital_index(s: &State, owner: i32) -> (usize, bool) {
    let mut best = i32::MAX;
    let mut idx = 0usize;
    let mut exists = false;
    for i in 0..s.n {
        if s.cities.owner[i] == owner && s.tiles.capital[i] == owner {
            exists = true;
            if s.cities.order[i] < best {
                best = s.cities.order[i];
                idx = i;
            }
        }
    }
    (idx, exists)
}

pub fn connections(s: &mut State, owner: i32, g: &Geom, native: bool) {
    let n = s.n;
    let (cap, exists) = capital_index(s, owner);
    let t = &s.tiles;
    let po = (owner - 1) as usize;
    let friendly: Vec<bool> = (0..n)
        .map(|i| t.owner[i] == 0 || t.owner[i] == owner || (native && s.players.peace[po][s.pclip(t.owner[i])]))
        .collect();
    let nodes: Vec<bool> = (0..n)
        .map(|i| (t.road[i] || t.structure[i] == 48 || t.structure[i] == 8 || (s.cities.owner[i] > 0 && friendly[i])) && friendly[i])
        .collect();
    let ports: Vec<usize> = (0..n).filter(|&i| t.structure[i] == 8).collect();
    let water: Vec<bool> = (0..n).map(|i| t.terrain[i] == 1 || t.terrain[i] == 2).collect();
    // port_edges[p] = ports reachable from port p through friendly water within the reference's distance rule.
    let mut port_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    for &p in &ports {
        seen.fill(false);
        seen[p] = true;
        stack.clear();
        stack.push(p);
        while let Some(x) = stack.pop() {
            if g.d(p, x) < 4 {
                for &y in &g.adj[x] {
                    let y = y as usize;
                    if !seen[y] && water[y] && friendly[y] {
                        seen[y] = true;
                        stack.push(y);
                    }
                }
            }
        }
        for &q in &ports {
            if seen[q] && g.d(p, q) <= 4 {
                port_edges[p].push(q);
            }
        }
    }
    let mut reached = vec![false; n];
    if exists {
        reached[cap] = true;
        stack.clear();
        stack.push(cap);
        while let Some(x) = stack.pop() {
            for &y in &g.adj[x] {
                let y = y as usize;
                if !reached[y] && nodes[y] {
                    reached[y] = true;
                    stack.push(y);
                }
            }
            for &q in &port_edges[x] {
                if !reached[q] {
                    reached[q] = true;
                    stack.push(q);
                }
            }
        }
    }
    let mut amount = vec![0i32; n];
    let mut total = 0i32;
    for c in 0..n {
        let own = s.cities.owner[c] == owner && c != cap && exists;
        if own {
            let delta = reached[c] as i32 - s.cities.connected[c] as i32;
            amount[c] = delta;
            total += delta;
            s.cities.connected[c] = reached[c];
        }
    }
    amount[cap] += total;
    let pop_owner = if native { owner } else { s.current_player };
    add_population_vector(s, &amount, pop_owner, native);
}

pub fn monument_conditions(s: &State, owner: i32, r: &Rules, g: &Geom, out: &mut [bool; 128]) {
    let c = &s.cities;
    let (mut connected, mut level5) = (0usize, false);
    for i in 0..s.n {
        if c.owner[i] == owner {
            connected += c.connected[i] as usize;
            level5 |= c.level[i] >= 5;
        }
    }
    let bits = monument_bits(s, owner, r, g, connected, level5);
    for k in 0..128 {
        out[k] = bits >> k & 1 == 1;
    }
}

/// monument_conditions as a bitset, given the owner's connected-city count and whether a city of theirs has level 5.
pub fn monument_bits(s: &State, owner: i32, r: &Rules, g: &Geom, connected: usize, level5: bool) -> u128 {
    let p = &s.players;
    let pi = (owner - 1) as usize;
    let tech = &p.tech[pi];
    let all_tech = bits128(tech) | !r.tech_exists_bits == !0;
    let explorer = g.corners.iter().all(|&t| s.tiles.explored[pi][t]) && s.fog;
    let bit = |k: u32, v: bool| (v as u128) << k;
    // the Network task (Grand Bazaar) is enabled by the first city connection, not by the Roads tech
    bit(23, tech[21] && p.pacifist[pi] >= 5) | bit(24, all_tech) | bit(25, connected >= 5)
        | bit(26, tech[5] && p.stars[pi] >= 100) | bit(27, p.kills[pi] >= 10) | bit(28, level5) | bit(29, explorer)
}

pub fn remember_monuments(s: &mut State, r: &Rules, g: &Geom) {
    // one pass over the cities for every seat's connected counts and level-5 flags
    let c = &s.cities;
    let np = s.np as i32;
    let (mut connected, mut level5) = ([0usize; crate::state::MAXP], [false; crate::state::MAXP]);
    for i in 0..s.n {
        let o = c.owner[i];
        if (1..=np).contains(&o) {
            connected[(o - 1) as usize] += c.connected[i] as usize;
            level5[(o - 1) as usize] |= c.level[i] >= 5;
        }
    }
    for owner in 1..=np {
        let pi = (owner - 1) as usize;
        let earned = monument_bits(s, owner, r, g, connected[pi], level5[pi]);
        let m = &mut s.players.monuments[pi];
        for k in 0..128 {
            m[k] |= earned >> k & 1 == 1;
        }
    }
}

/// Per-call inputs of structures.build_mask shared by every tile (BuildCtx::prepare).
pub struct BuildCtx {
    /// economic_tiles(s), filled by prepare(), and the same tiles as a list
    pub econ: Vec<bool>,
    pub econ_list: Vec<u16>,
    own_cities: Vec<usize>,
    /// structures present in the territory of each own city (bit = structure kind)
    present: Vec<u128>,
    unlocked: u128,
    monuments: u128,
}

const ROAD: u128 = 1 << 3;
const BRIDGE: u128 = 1 << 48;

impl BuildCtx {
    pub fn road_unlocked(&self) -> bool {
        self.unlocked & ROAD != 0
    }

    pub fn new(n: usize) -> BuildCtx {
        BuildCtx { econ: vec![false; n], econ_list: Vec::with_capacity(n), own_cities: Vec::with_capacity(n), present: Vec::with_capacity(n), unlocked: 0,
                   monuments: 0 }
    }

    /// Everything of build_mask that does not depend on the tile, including economic_tiles(s) into `econ` (one
    /// pass over each own city's territory row gives both the economy and the structures present per city).
    pub fn prepare(&mut self, s: &State, r: &Rules, g: &Geom) {
        let n = s.n;
        let (t, c, p) = (&s.tiles, &s.cities, &s.players);
        let cp = s.current_player;
        let pi = s.p();
        let rs = &r.structures;
        self.own_cities.clear();
        self.present.clear();
        let econ = &mut self.econ[..n];
        econ.fill(false);
        let mut tribe_has = 0u128;
        let (mut connected, mut level5) = (0usize, false);
        for city in 0..n {
            if c.owner[city] != cp {
                continue;
            }
            connected += c.connected[city] as usize;
            level5 |= c.level[city] >= 5;
            let mut pr = 0u128;
            let row = &c.territory[city * n..city * n + n];
            for tile in 0..n {
                if row[tile] {
                    pr |= 1u128 << t.structure[tile];
                    econ[tile] = true;
                }
            }
            tribe_has |= pr;
            self.own_cities.push(city);
            self.present.push(pr);
        }
        self.econ_list.clear();
        for tile in 0..n {
            // GameLogicData.CanBuild: the tile itself must be the player's, whatever the city rows say
            econ[tile] = econ[tile] && crate::economy::unit_allows_work(s, tile) && t.owner[tile] == cp;
            if econ[tile] {
                self.econ_list.push(tile as u16);
            }
        }
        let mut tech_ok = 0u128;
        let mut known = bits128(&p.tech[pi]);
        while known != 0 {
            tech_ok |= rs.tech_unlocks[known.trailing_zeros() as usize];
            known &= known - 1;
        }
        self.unlocked = tech_ok & rs.static_ok & rs.cost_le.le(p.stars[pi]) & !(rs.per_tribe_bits & tribe_has);
        let earned = monument_bits(s, cp, r, g, connected, level5) | bits128(&p.monuments[pi]);
        // ImprovementAbility Unique (PlayerState.builtUniqueImprovements): a monument is blocked only by the player
        // having built it, not by one standing in a captured city's territory
        self.monuments = earned & !bits128(&p.built[pi]);
    }
}

/// Buildable structure kinds on `tile` as a bitset: one row of structures.build_mask (native profile), with each
/// of its four terms evaluated for all 128 kinds at once as bitset algebra:
///   (a) unlocked & terrain & empty & adjacency & city room   (b) neutral road   (c) the tile's resource building
///   (d) earned monuments on free land; then the bridge span test.
pub fn build_row(s: &State, r: &Rules, g: &Geom, ctx: &BuildCtx, tile: usize) -> u128 {
    let (t, c, p) = (&s.tiles, &s.cities, &s.players);
    let cp = s.current_player;
    let pi = s.p();
    let rs = &r.structures;
    let terrain = rs.terrain_bits[t.terrain[tile].clamp(0, 15) as usize];
    let b = (if neutral_road(s, r, ctx, tile) { ROAD } else { 0 }) | (if neutral_bridge(s, r, ctx, tile) { BRIDGE } else { 0 });
    if !ctx.econ[tile] {
        return b; // every other term carries econ[tile]
    }
    let free = t.structure[tile] == 0;
    let mut adjacency = rs.no_adjacent_bits;
    for &j in &g.adj[tile] {
        let j = j as usize;
        if t.owner[j] == cp {
            adjacency |= rs.adjacent_rev[t.structure[j] as usize];
        }
    }
    // a per-city kind has room iff some own city holding this tile lacks it
    let mut holders = 0usize;
    let mut present_all = !0u128;
    for (k, &city) in ctx.own_cities.iter().enumerate() {
        if c.territory[city * s.n + tile] {
            holders += 1;
            present_all &= ctx.present[k];
        }
    }
    let room = if holders > 0 { !rs.per_city_bits | !present_all } else { 0 };
    // TileData.HasRoad is true on every City tile, villages included, so no road goes there
    let road_ok = !t.road[tile] && c.owner[tile] == 0 && t.structure[tile] != 1;
    let empty = (if free { !ROAD } else { 0 }) | (if road_ok { ROAD } else { 0 });
    let a = ctx.unlocked & terrain & empty & adjacency & room;
    let (res_tech, res_build) = match t.resource[tile] { 2 => (8, 5), 5 => (23, 21), 7 => (0, 37), _ => (0, 0) };
    let resource = free && res_build > 0 && p.tech[pi][res_tech] && p.stars[pi] >= rs.cost[res_build];
    let d = if free { terrain & ctx.monuments } else { 0 };
    let mut v = a | b | ((resource as u128) << res_build) | d;
    if v & BRIDGE != 0 && bridge_span(s, tile) == 0 {
        v &= !BRIDGE;
    }
    v
}

/// TileData.GetPossibleBridgeDirection (from the true terrain): 1 for a north-south span (land north and
/// south), 2 for east-west (land east and west), 0 for none; north-south wins when both hold.
#[inline]
pub fn bridge_span(s: &State, tile: usize) -> u8 {
    let size = s.size;
    let (x, y) = (tile % size, tile / size);
    let land = |i: usize| s.tiles.terrain[i] != 1 && s.tiles.terrain[i] != 2;
    if y > 0 && y + 1 < size && land(tile - size) && land(tile + size) {
        1
    } else if x > 0 && x + 1 < size && land(tile - 1) && land(tile + 1) {
        2
    } else {
        0
    }
}

/// GameLogicData.CanBuild (a Bridge is Freelance like a road): an unlocked bridge on a neutral, explored,
/// structure-less water tile with a span, free of other players' visible units.
#[inline]
pub fn neutral_bridge(s: &State, r: &Rules, ctx: &BuildCtx, tile: usize) -> bool {
    let t = &s.tiles;
    ctx.unlocked & BRIDGE != 0 && t.owner[tile] == 0 && t.explored[s.p()][tile] && t.structure[tile] == 0
        && crate::economy::unit_allows_work(s, tile)
        && r.structures.terrain_bits[t.terrain[tile].clamp(0, 15) as usize] & BRIDGE != 0 && bridge_span(s, tile) != 0
}

/// Term (b) of build_row: an unlocked road on a neutral, explored, road-less tile whose terrain allows roads.
#[inline]
pub fn neutral_road(s: &State, r: &Rules, ctx: &BuildCtx, tile: usize) -> bool {
    let t = &s.tiles;
    ctx.unlocked & ROAD != 0 && t.owner[tile] == 0 && t.explored[s.p()][tile] && !t.road[tile]
        && t.structure[tile] != 1
        && crate::economy::unit_allows_work(s, tile)
        && r.structures.terrain_bits[t.terrain[tile].clamp(0, 15) as usize] & ROAD != 0
}

/// build_mask into out[t*128 + kind] (structures.build_mask, native profile).
pub fn build_mask(s: &State, r: &Rules, g: &Geom, out: &mut [bool]) {
    let mut ctx = BuildCtx::new(s.n);
    ctx.prepare(s, r, g);
    for tile in 0..s.n {
        let bits = build_row(s, r, g, &ctx, tile);
        for kind in 0..128 {
            out[tile * 128 + kind] = bits >> kind & 1 == 1;
        }
    }
}

/// One entry of build_mask (step validation).
pub fn build_legal(s: &State, r: &Rules, g: &Geom, ctx: &mut BuildCtx, tile: usize, kind: usize) -> bool {
    ctx.prepare(s, r, g);
    build_row(s, r, g, ctx, tile) >> kind & 1 == 1
}

pub fn build(s: &mut State, target: usize, kind: usize, r: &Rules, g: &Geom) {
    let before = population_contribution(s, r, g);
    let rs = &r.structures;
    let road = kind == 3;
    let pi = s.p();
    if !road {
        s.tiles.structure[target] = kind as i32;
        s.tiles.structure_level[target] = 1;
        s.tiles.structure_turn[target] = s.turn;
    }
    s.tiles.road[target] |= road;
    s.players.stars[pi] -= rs.cost[kind].max(0);
    if kind == 3 || kind == 8 || kind == 48 {
        connections(s, s.current_player, g, true);
    }
    let score = if temple(kind as i32) { 100 } else { rs.score[kind] };
    let owner = s.tiles.owner[target];
    if owner > 0 {
        let o = s.pclip(owner);
        s.players.score[o] = s.players.score[o].wrapping_add(score);
    }
    s.players.built[pi][kind] |= score > 0;
    adjust_building_population(&before, s, r, g);
}

pub fn destroy_mask(s: &State, out: &mut [bool]) {
    economic_tiles(s, out);
    let econ = out[..s.n].to_vec();
    destroy_mask_with(s, &econ, out);
}

/// Structures Destroy removes (ImprovementDataExtensions.IsDestructable): not cities, ruins or lighthouses.
#[inline]
pub fn destructible(k: i32) -> bool {
    k > 0 && k != 1 && k != 2 && k != 47
}

/// destroy_mask given economic_tiles(s).
pub fn destroy_mask_with(s: &State, econ: &[bool], out: &mut [bool]) {
    let tech3 = s.players.tech[s.p()][3];
    for t in 0..s.n {
        out[t] = econ[t] && destructible(s.tiles.structure[t]) && tech3;
    }
}

pub fn destroy(s: &mut State, target: usize, r: &Rules, g: &Geom) {
    let before = population_contribution(s, r, g);
    let kind = s.tiles.structure[target];
    let score = score_of(s, r, target);
    let owner = s.tiles.owner[target];
    s.tiles.structure[target] = 0;
    s.tiles.structure_level[target] = 0;
    s.tiles.structure_turn[target] = 0;
    s.tiles.road[target] = s.tiles.road[target] && kind != 3 && kind != 48;
    if owner > 0 {
        let o = s.pclip(owner);
        s.players.score[o] = s.players.score[o].wrapping_sub(score);
    }
    adjust_building_population(&before, s, r, g);
    if kind == 3 || kind == 8 || kind == 48 {
        connections(s, s.current_player, g, true);
    }
}

pub fn start_turn(s: &mut State, owner: i32) {
    let n = s.n;
    let mut gain = 0i32;
    for i in 0..n {
        let t = &mut s.tiles;
        if temple(t.structure[i]) && t.owner[i] == owner {
            let expected = (1 + (s.turn - t.structure_turn[i]).div_euclid(2)).clamp(1, 5);
            let new = expected.max(t.structure_level[i]);
            gain += new - t.structure_level[i];
            t.structure_level[i] = new;
        }
    }
    let o = (owner - 1) as usize;
    s.players.score[o] = s.players.score[o].wrapping_add(gain * 100);
}

pub fn income(s: &State, owner: i32, r: &Rules, g: &Geom) -> i32 {
    let n = s.n;
    let (t, c, u) = (&s.tiles, &s.cities, &s.units);
    // the territory bonus is the markets' value; without a market it is zero
    let markets = t.structure.contains(&50);
    let market_value: Vec<i32> = if markets {
        let levels = production_levels(s, r, g);
        (0..n).map(|i| if t.structure[i] == 50 { levels[i] * r.structures.stars[50] } else { 0 }).collect()
    } else {
        Vec::new()
    };
    let mut total = 0i32;
    if markets {
        // ResourceDataUtils.CalculateIncomeFor sums TileDataExtensions.CalculateWork over every tile: each owned
        // market pays on its own, whatever happens to its city (a siege or negative progress stops only the city)
        total += (0..n).filter(|&i| t.owner[i] == owner).map(|i| market_value[i]).sum::<i32>();
    }
    for city in 0..n {
        if c.owner[city] != owner {
            continue;
        }
        // any unit of another player on the city stops its work, a hidden one included
        let siege = (u.owner[city] != 0 && u.owner[city] != c.owner[city]) || c.riot[city];
        if siege {
            continue;
        }
        let mut production = c.production[city] + (t.capital[city] == c.owner[city] && t.capital[city] > 0) as i32;
        if c.progress[city] < 0 {
            production = (production + c.progress[city]).max(0);
        }
        total += production;
    }
    let p = (owner - 1) as usize;
    let np = s.np;
    let embassy_count = (0..np).map(|k| s.players.embassy[p][k] as i32).sum::<i32>()
        + (0..np).map(|k| s.players.embassy[k][p] as i32).sum::<i32>();
    // embassies exist only in two-seat games
    total + embassy_count * if np == 2 && s.players.peace[p][1 - p] { 4 } else { 2 }
}

pub fn _unused(s: &State) {
    let mut v = vec![false; s.n];
    in_territory(s, &mut v);
    let _ = add_population;
}
}
mod discovery {

use crate::geom::Geom;
use crate::hash::hash_word;
use crate::state::State;

pub fn explorer(s: &State, source: usize, g: &Geom) -> Vec<bool> {
    let n = s.n;
    let p = s.p();
    let initial = &s.tiles.explored[p];
    let tech = &s.players.tech[p];
    let mut visible: Vec<bool> = initial.clone();
    let mut current = source;
    let mut previous = -1i32;
    let mut score = vec![0i32; n];
    let mut spread = vec![0i32; n];
    for _ in 0..12 {
        for t in 0..n {
            if visible[t] {
                score[t] = 1000;
            } else {
                let mut count = 0;
                let mut light = false;
                for j in 0..n {
                    if g.d(t, j) <= 2 && !visible[j] {
                        count += 1;
                        if s.tiles.structure[j] == 47 {
                            light = true;
                        }
                    }
                }
                let count = count.clamp(1, 4);
                score[t] = 183 - count * 10 - if light { 33 } else { 0 };
            }
        }
        for _ in 0..3 {
            for t in 0..n {
                let mut best = 1000;
                for &j in &g.adj[t] {
                    best = best.min(score[j as usize] + 100);
                }
                spread[t] = score[t].min(best);
            }
            std::mem::swap(&mut score, &mut spread);
        }
        let mut allowed = [false; 8];
        let mut values = [10000i32; 8];
        let mut best = 10000i32;
        for k in 0..8 {
            if !g.nb_valid[current * 8 + k] {
                continue;
            }
            let idx = g.nb[current * 8 + k] as usize;
            let terrain = s.tiles.terrain[idx];
            let passable = terrain != 0 && (terrain != 4 || tech[20]) && (terrain != 1 || tech[13]) && (terrain != 2 || tech[14]);
            allowed[k] = !visible[idx] || passable;
            if allowed[k] {
                values[k] = score[idx] + (idx as i32 == previous) as i32;
                best = best.min(values[k]);
            }
        }
        let candidates: Vec<usize> = (0..8).filter(|&k| allowed[k] && values[k] == best).collect();
        let count = candidates.len() as u32;
        if count == 0 {
            break;
        }
        let draw = hash_word(s.seed, current as u32) % count;
        let target = g.nb[current * 8 + candidates[draw as usize]] as usize;
        if target == current {
            break;
        }
        for j in 0..n {
            if g.d(target, j) <= 1 {
                visible[j] = true;
            }
        }
        previous = current as i32;
        current = target;
    }
    (0..n).map(|t| visible[t] && !initial[t]).collect()
}
}
mod diplomacy {

use crate::economy::discover;
use crate::geom::Geom;
use crate::military::remove;
use crate::rules::Rules;
use crate::state::State;
use crate::structures::connections;

#[inline]
pub fn protected(s: &State, owner: i32, other: i32) -> bool {
    let (p, q) = (s.pclip(owner), s.pclip(other));
    other > 0 && (owner == other || s.players.peace[p][q] || s.players.truce_breaker[p][q] != 0)
}

/// Opponent capital tile (first match) and whether one exists; shared by legal_masks and decode.
pub fn opponent_capital(s: &State) -> (usize, bool) {
    if s.np != 2 {
        return (0, false);
    }
    let other = 3 - s.current_player;
    for i in 0..s.n {
        if s.tiles.capital[i] == other && s.cities.owner[i] == other {
            return (i, true);
        }
    }
    (0, false)
}

/// diplomacy.legal for Action(3, source, target, ability).
pub fn legal(s: &State, source: i32, target: i32, ability: i32) -> bool {
    legal_with(s, source, target, ability, opponent_capital(s))
}

/// diplomacy.legal given opponent_capital(s).
pub fn legal_with(s: &State, source: i32, target: i32, ability: i32, (capital, has_capital): (usize, bool)) -> bool {
    if s.np != 2 {
        return false;
    }
    let p = s.p();
    let q = (2 - s.current_player) as usize;
    let pl = &s.players;
    let other = q as i32 + 1;
    let eligible = source == other && pl.known[p][q] && pl.killed_turn[q] == 0 && pl.resigned_turn[q] == 0;
    eligible
        && ((ability == 18 && pl.peace[p][q])
            || (ability == 19 && pl.peace_offer[p][q] && (target == 0 || target == 1))
            || (ability == 20 && pl.tech[p][38] && !pl.embassy[p][q] && has_capital && target == capital as i32
                && pl.stars[p] >= 5 && s.turn * 2 + p as i32 >= pl.hostile_until[p])
            || (ability == 21 && pl.tech[p][7] && !pl.peace[p][q] && !pl.peace_offer[q][p] && pl.truce_breaker[p][q] == 0)
            || (ability == 22 && pl.embassy[p][q]))
}

pub fn hostile(s: &mut State) {
    let p = s.p();
    s.players.embassy = Default::default();
    s.players.attacked[p] = true;
    s.players.peace_offer = Default::default();
    s.players.hostile_until[p] = s.turn * s.np as i32 + p as i32 + 2;
}

pub fn capital_vision(s: &mut State) {
    let n = s.n;
    let p = s.p();
    if !s.players.tech[p][38] {
        return; // discover of an empty reveal changes nothing
    }
    // the capitals of every met seat (with two seats: the opponent's, once met)
    let met: Vec<i32> = (0..s.np).filter(|&q| q != p && s.players.known[p][q]).map(|q| q as i32 + 1).collect();
    if met.is_empty() {
        return;
    }
    let reveal: Vec<bool> = (0..n).map(|t| met.contains(&s.tiles.capital[t])).collect();
    discover(s, &reveal, s.current_player);
}

pub fn start_turn(s: &mut State, owner: i32) {
    for a in 0..s.np {
        for b in 0..s.np {
            if s.players.truce_breaker[a][b] == owner {
                s.players.truce_breaker[a][b] = 0;
            }
        }
    }
}

pub fn execute(s: &mut State, target: i32, ability: i32, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let p = s.p();
    let q = (2 - cp) as usize;
    match ability {
        18 => {
            s.players.peace[p][q] = false;
            s.players.peace[q][p] = false;
            s.players.truce_breaker[p][q] = cp;
            s.players.truce_breaker[q][p] = cp;
            let doomed: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] == cp && s.tiles.owner[i] == q as i32 + 1).collect();
            for i in doomed {
                remove(s, i, r, None, true);
            }
            for i in 0..n {
                if s.units.owner[i] == cp {
                    s.units.moved[i] = true;
                    s.units.attacked[i] = true;
                }
            }
        }
        19 => {
            let accepted = target == 1;
            s.players.peace[p][q] = accepted;
            s.players.peace[q][p] = accepted;
            s.players.peace_offer[p][q] = false;
            s.players.peace_offer[q][p] = false;
        }
        20 => {
            s.players.embassy[p][q] = true;
            s.players.stars[p] -= 5;
            s.players.known[q][p] = true;
            let t = target.clamp(0, n as i32 - 1) as usize;
            let reveal: Vec<bool> = (0..n).map(|i| g.d(t, i) <= 1).collect();
            discover(s, &reveal, cp);
        }
        21 => {
            s.players.peace_offer[q][p] = true;
        }
        _ => {
            s.players.embassy[p][q] = false;
        }
    }
    connections(s, cp, g, true);
    connections(s, 3 - cp, g, true);
}
}
mod kernels {

use crate::geom::Geom;
use crate::rules::{Rules, CARRY, CREEP, FLOAT, HIDE, INFILTRATE};
use crate::state::State;

pub const INF: i32 = 1_000_000;

/// Reference f32 damage formula with Rust's half-away-from-zero rounding; (attack, defense, attack*0.5).
#[inline]
pub fn combat(aa: f32, ah: f32, am: f32, dd: f32, dh: f32, dm: f32, db: f32) -> (f32, f32, f32) {
    let af = aa * (ah / am.max(1.0));
    let df = dd * (dh / dm.max(1.0)) * db;
    let total = af + df;
    let divisor = if total > 0.0 { total } else { 1.0 };
    let attack = ((if total > 0.0 { (af / divisor) * aa * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    let defense = ((if dh > attack && total > 0.0 { (df / divisor) * dd * 4.5 } else { 0.0 }) + 0.000001f32 + 0.5).floor();
    (attack, defense, attack * 0.5)
}

#[inline]
pub fn max_health(s: &State, r: &Rules, i: usize) -> f32 {
    let u = &s.units;
    let kind = if u.passenger[i] > 0 { u.passenger[i] } else { u.kind[i] };
    r.unit_health[(kind as usize).min(127)] + (u.veteran[i] as i32 * 5) as f32
}

pub fn research_cost(s: &State, r: &Rules, k: usize) -> i32 {
    let cp = s.current_player;
    let cities = s.cities.owner.iter().filter(|&&o| o == cp).count() as i32;
    research_cost_with(s, r, k, cities)
}

/// research_cost given the mover's city count.
#[inline]
pub fn research_cost_with(s: &State, r: &Rules, k: usize, cities: i32) -> i32 {
    let cost = 4 + cities * r.tech_tier[k];
    // GameLogicData.GetTechPrice: Literacy multiplies by 0.66666f
    if s.players.tech[s.p()][22] { ((cost as f32) * 0.66666f32).ceil() as i32 } else { cost }
}

pub fn research_mask(s: &State, r: &Rules, out: &mut [bool]) {
    let active = !s.pending_reward() && !s.terminated && !s.truncated;
    let bits = if active { research_bits(s, r) } else { 0 };
    for k in 0..128 {
        out[k] = bits >> k & 1 == 1;
    }
}

/// research_mask without its `active` gate, as a bitset: techs reachable from a discovered one, not discovered,
/// existing and affordable (one city count, predecessor bitsets).
pub fn research_bits(s: &State, r: &Rules) -> u128 {
    let p = s.p();
    let discovered = &s.players.tech[p];
    let stars = s.players.stars[p];
    let known = crate::rules::bits128(discovered);
    let mut cand = 0u128;
    let mut rest = known;
    while rest != 0 {
        cand |= r.tech_succ[rest.trailing_zeros() as usize];
        rest &= rest - 1;
    }
    cand &= !known & r.tech_exists_bits;
    if cand == 0 {
        return 0;
    }
    let cp = s.current_player;
    let cities = s.cities.owner.iter().filter(|&&o| o == cp).count() as i32;
    let mut out = 0u128;
    while cand != 0 {
        let k = cand.trailing_zeros() as usize;
        cand &= cand - 1;
        out |= ((research_cost_with(s, r, k, cities) <= stars) as u128) << k;
    }
    out
}

pub struct MovementData {
    pub passable: Vec<bool>,
    pub terminal: Vec<bool>,
    pub road: Vec<bool>,
    pub enemy: Vec<bool>,
    pub budget: i32,
}

impl MovementData {
    pub fn new(n: usize) -> MovementData {
        MovementData { passable: vec![false; n], terminal: vec![false; n], road: vec![false; n], enemy: vec![false; n], budget: 0 }
    }
}

pub fn movement_data(s: &State, source: usize, r: &Rules, g: &Geom) -> MovementData {
    let mut md = MovementData::new(s.n);
    movement_data_into(s, source, r, g, &mut md);
    md
}

/// kernels.movement_data into reusable buffers.
pub fn movement_data_into(s: &State, source: usize, r: &Rules, g: &Geom, md: &mut MovementData) {
    let n = s.n;
    let (t, u) = (&s.tiles, &s.units);
    let owner = u.owner[source];
    let p = s.pclip(owner);
    let tech = &s.players.tech[p];
    let kind = u.kind[source] as usize;
    let aquatic = r.skills[kind][FLOAT];
    let creep = r.skills[kind][CREEP];
    let (passable, enemy, road, terminal) = (&mut md.passable, &mut md.enemy, &mut md.road, &mut md.terminal);
    for i in 0..n {
        let bridge = t.structure[i] == 48;
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        let terrain_ok = ((t.terrain[i] != 1 || tech[10]) && (t.terrain[i] != 2 || tech[13] || tech[25]) && (t.terrain[i] != 4 || tech[20])) || bridge;
        let own_port = t.structure[i] == 8 && t.owner[i] == owner;
        let land_access = !water || bridge || own_port;
        let hidden_enemy = u.invisible[i] && u.owner[i] != owner;
        passable[i] = t.explored[p][i] && terrain_ok && (aquatic || land_access) && (u.owner[i] == 0 || u.owner[i] == owner || hidden_enemy);
        let peaceful = s.players.peace[p][s.pclip(u.owner[i])];
        enemy[i] = u.owner[i] != 0 && u.owner[i] != owner && !u.invisible[i] && !peaceful;
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || s.players.peace[p][s.pclip(t.owner[i])];
        road[i] = (t.road[i] || t.structure[i] == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
    }
    let no_zoc = creep || r.skills[kind][INFILTRATE] || u.invisible[source];
    for i in 0..n {
        let zoc = !no_zoc && g.adj[i].iter().any(|&j| enemy[j as usize]);
        let water = t.terrain[i] == 1 || t.terrain[i] == 2;
        let own_port = t.structure[i] == 8 && t.owner[i] == owner;
        terminal[i] = zoc || t.terrain[i] == 4 || (t.terrain[i] == 5 && !road[i] && !creep) || (own_port && !aquatic) || (r.skills[kind][CARRY] && !water);
    }
    md.budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
}

// Unit-independent movement terms of one owner, one byte per tile (movement_flags).
pub const MV_PASS: u8 = 1; // explored, terrain allowed by tech (or bridge), no blocking unit
pub const MV_LAND: u8 = 2; // land access: not water, or a bridge, or an own port
pub const MV_ROAD: u8 = 4; // friendly road network tile
pub const MV_ZOC: u8 = 8; // next to a visible, non-peaceful, explored enemy unit
pub const MV_MOUNTAIN: u8 = 16;
pub const MV_FOREST: u8 = 32;
pub const MV_OWN_PORT: u8 = 64;
pub const MV_WATER: u8 = 128;

/// kernels.movement_data's per-tile terms for units of `owner`, which only the unit's skills (float, creep,
/// carry, zoc immunity) and budget specialise: passable = PASS && (float || LAND); terminal = (ZOC && !no_zoc)
/// || MOUNTAIN || (FOREST && !ROAD && !creep) || (OWN_PORT && !float) || (carry && !WATER).
pub fn movement_flags(s: &State, owner: i32, g: &Geom, flags: &mut [u8]) {
    let n = s.n;
    let (t, u) = (&s.tiles, &s.units);
    let p = s.pclip(owner);
    let tech = &s.players.tech[p];
    let peace = &s.players.peace[p];
    let explored = &t.explored[p];
    for i in 0..n {
        let (terrain, structure) = (t.terrain[i], t.structure[i]);
        let bridge = structure == 48;
        let water = terrain == 1 || terrain == 2;
        let terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20])) || bridge;
        let own_port = structure == 8 && t.owner[i] == owner;
        let uo = u.owner[i];
        let hidden_enemy = u.invisible[i] && uo != owner;
        let pass = explored[i] && terrain_ok && (uo == 0 || uo == owner || hidden_enemy);
        let friendly_road = t.owner[i] == 0 || t.owner[i] == owner || peace[s.pclip(t.owner[i])];
        let road = (t.road[i] || structure == 1 || s.cities.owner[i] != 0 || bridge) && friendly_road;
        flags[i] = (pass as u8) * MV_PASS | ((!water || bridge || own_port) as u8) * MV_LAND | (road as u8) * MV_ROAD
            | ((terrain == 4) as u8) * MV_MOUNTAIN | ((terrain == 5) as u8) * MV_FOREST | (own_port as u8) * MV_OWN_PORT
            | (water as u8) * MV_WATER;
    }
    for i in 0..n {
        let uo = u.owner[i];
        // TileData.GetMovementCost reads the raw neighbours: unseen enemies exert zone of control too
        let enemy = uo != 0 && uo != owner && !u.invisible[i] && !peace[s.pclip(uo)];
        if enemy {
            for &j in &g.adj[i] {
                flags[j as usize] |= MV_ZOC;
            }
        }
    }
}

/// PathFinder.IsTileAccessible: a Hide unit may pass a tile held by another player's unit (it never
/// ends there). True when tile `i` stops `owner`'s units only because of such a unit: explored, terrain allowed.
#[inline]
pub fn hide_passable(s: &State, owner: i32, i: usize) -> bool {
    let (t, u) = (&s.tiles, &s.units);
    let tech = &s.players.tech[s.pclip(owner)];
    let terrain = t.terrain[i];
    let terrain_ok = ((terrain != 1 || tech[10]) && (terrain != 2 || tech[13] || tech[25]) && (terrain != 4 || tech[20]))
        || t.structure[i] == 48;
    u.owner[i] != 0 && u.owner[i] != owner && t.explored[s.pclip(owner)][i] && terrain_ok
}

/// PathFinder.IsTileAccessible: any unit may pass a tile held by a peace partner's unit.
#[inline]
pub fn peace_unit(s: &State, owner: i32, i: usize) -> bool {
    let o = s.units.owner[i];
    o != 0 && o != owner && s.players.peace[s.pclip(owner)][s.pclip(o)]
}

/// TileData.GetMovementCost: entering a forest ends the move unless the step runs road to road. `cand` is the step's cost so far.
#[inline]
pub fn forest_step(creep: bool, road_edge: bool, dest_flags: u8, cand: i32, budget: i32) -> i32 {
    if !creep && !road_edge && dest_flags & MV_FOREST != 0 { cand.max(budget) } else { cand }
}

/// PathFinder.IsTileAccessible: without water movement (Fishing), a bridge cannot be entered across its
/// span: not straight from the east or west onto a north-south bridge, nor straight from north or south onto an
/// east-west one (diagonal steps are allowed).
#[inline]
pub fn bridge_blocks(s: &State, owner: i32, from: usize, to: usize) -> bool {
    if s.tiles.structure[to] != 48 || s.players.tech[s.pclip(owner)][10] {
        return false;
    }
    let size = s.size;
    let (dx, dy) = ((to % size) as i32 - (from % size) as i32, (to / size) as i32 - (from / size) as i32);
    match crate::structures::bridge_span(s, to) {
        1 => dy == 0,
        2 => dx == 0,
        _ => false,
    }
}

/// Reusable buffers for movement_targets.
pub struct MoveScratch {
    pub flags: Vec<u8>,
    pub dist: Vec<i32>,
    stamp: Vec<u32>,
    epoch: u32,
    frontier: Vec<(u16, i32)>,
    next: Vec<u16>,
    touched: Vec<u16>,
}

impl MoveScratch {
    pub fn new(n: usize) -> MoveScratch {
        MoveScratch { flags: vec![0; n], dist: vec![INF; n], stamp: vec![0; n], epoch: 0, frontier: Vec::with_capacity(n),
                      next: Vec::with_capacity(n), touched: Vec::with_capacity(n) }
    }
}

/// Destination tiles of the unit on `source` (kernels.movement_targets without its `active` gate), given
/// `ms.flags` = movement_flags of the unit's owner. kernels.movement_targets runs 8 synchronous relaxation sweeps
/// over every tile; this runs the same 8 synchronous rounds (Jacobi: every relaxation in round k reads the
/// distances of round k-1) over the frontier only, i.e. the tiles whose distance changed in the previous round,
/// which is exact: a tile that did not change contributes nothing new. Calls `emit(tile)` per destination.
pub fn reach(s: &State, source: usize, r: &Rules, g: &Geom, ms: &mut MoveScratch, mut emit: impl FnMut(usize)) {
    let u = &s.units;
    let owner = u.owner[source];
    let kind = u.kind[source] as usize;
    let sk = &r.skills[kind];
    let (aquatic, creep, carry) = (sk[FLOAT], sk[CREEP], sk[CARRY]);
    let no_zoc = creep || sk[INFILTRATE] || u.invisible[source];
    let budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
    let hide = sk[HIDE];
    // terminal: any of these bits set (plus FOREST without ROAD for non-creepers, handled below)
    let term_bits = (if no_zoc { 0 } else { MV_ZOC }) | MV_MOUNTAIN | (if aquatic { 0 } else { MV_OWN_PORT });
    let pass_need = if aquatic { MV_PASS } else { MV_PASS | MV_LAND };
    let MoveScratch { flags, dist, stamp, epoch, frontier, next, touched } = ms;
    let terminal = |f: u8| f & term_bits != 0 || (f & (MV_FOREST | MV_ROAD) == MV_FOREST && !creep) || (carry && f & MV_WATER == 0);
    touched.clear();
    frontier.clear();
    dist[source] = 0;
    touched.push(source as u16);
    frontier.push((source as u16, 0));
    for _ in 0..8 {
        if frontier.is_empty() {
            break;
        }
        *epoch = epoch.wrapping_add(1);
        if *epoch == 0 {
            stamp.fill(0);
            *epoch = 1;
        }
        next.clear();
        for &(j, dj) in frontier.iter() {
            let j = j as usize;
            let fj = flags[j];
            if dj >= budget || (j != source && terminal(fj)) {
                continue;
            }
            let road_j = fj & MV_ROAD != 0 && !creep;
            for &i in &g.adj[j] {
                let i = i as usize;
                let fi = flags[i];
                if fi & pass_need != pass_need && i != source
                    && !((hide || peace_unit(s, owner, i)) && (aquatic || fi & MV_LAND != 0) && hide_passable(s, owner, i)) {
                    continue;
                }
                if bridge_blocks(s, owner, j, i) {
                    continue;
                }
                let road_edge = road_j && fi & MV_ROAD != 0;
                let cand = forest_step(creep, road_edge, fi, dj + if road_edge { 1 } else { 2 }, budget);
                if cand < dist[i] {
                    if dist[i] == INF {
                        touched.push(i as u16);
                    }
                    dist[i] = cand;
                    if stamp[i] != *epoch {
                        stamp[i] = *epoch;
                        next.push(i as u16);
                    }
                }
            }
        }
        frontier.clear();
        frontier.extend(next.iter().map(|&i| (i, dist[i as usize])));
    }
    // PathFinder.IsAllowedToFinishOnTile: an Infiltrate unit never ends on an opponent's city, and no
    // unit ends on the city of a player it has peace or a cool-down with
    let po = s.pclip(owner);
    let finish_ok = |i: usize| {
        let co = s.cities.owner[i];
        co == 0 || co == owner
            || (!sk[INFILTRATE] && !s.players.peace[po][s.pclip(co)] && s.players.truce_breaker[po][s.pclip(co)] == 0)
    };
    for &i in touched.iter() {
        let i = i as usize;
        dist[i] = INF;
        let empty = u.owner[i] == 0 || (u.invisible[i] && u.owner[i] != owner);
        if empty && i != source && finish_ok(i) {
            emit(i);
        }
    }
}

/// Destination set of the unit on `source` (kernels.movement_targets).
pub fn movement_targets(s: &State, source: usize, r: &Rules, g: &Geom, ms: &mut MoveScratch, out: &mut [bool]) {
    let n = s.n;
    let u = &s.units;
    let kind = u.kind[source] as usize;
    out[..n].fill(false);
    let active = u.owner[source] == s.current_player && !u.moved[source] && r.unit_supported[kind] && !s.pending_reward()
        && !s.terminated && !s.truncated;
    if !active {
        return;
    }
    movement_flags(s, u.owner[source], g, &mut ms.flags);
    reach(s, source, r, g, ms, |i| out[i] = true);
}

pub fn update_exploration(s: &mut State, player: i32, g: &Geom) {
    let n = s.n;
    let p = (player - 1) as usize;
    let mut explored = vec![false; n];
    for t in 0..n {
        explored[t] = s.tiles.explored[p][t] || !s.fog;
    }
    for c in 0..n {
        if s.cities.owner[c] == player {
            let range = if s.tiles.capital[c] > 0 { 2 } else { s.cities.border[c] };
            g.for_ball(c, range, |t| {
                let corner = g.corners.contains(&t);
                if !corner || s.cities.owner[t] == player {
                    explored[t] = true;
                }
            });
        }
    }
    for i in 0..n {
        if s.units.owner[i] == player {
            let k = s.units.kind[i];
            let range = if s.tiles.terrain[i] == 4 || k == 42 || k == 38 || k == 39 { 2 } else { 1 };
            g.for_ball(i, range, |t| explored[t] = true);
        }
    }
    let mut gain = 0;
    for t in 0..n {
        if explored[t] && !s.tiles.explored[p][t] {
            gain += 5;
        }
        s.tiles.explored[p][t] = explored[t];
    }
    s.players.score[p] = s.players.score[p].wrapping_add(gain);
}

/// Packed public view: kernels.observe(state, player) fields in sorted-key order, int16 (cpu_engine.pack_view).
/// Games with more than two seats add "alive" (first in sorted order) and give the five relation fields one entry
/// per seat (view_fields / view_width).
pub const VIEW_FIELDS: [&str; 40] = [
    "capital", "detect", "embassies_received", "embassies_sent", "explored", "known_players", "own_built",
    "own_city_connected", "own_city_level", "own_city_population", "own_city_production", "own_city_progress",
    "own_city_rewards", "own_city_riot", "own_monuments", "own_unit_attacked", "own_unit_boosted", "own_unit_frozen",
    "own_unit_home", "own_unit_invisible", "own_unit_kills", "own_unit_moved", "own_unit_passenger", "own_unit_poison",
    "owner", "peace", "peace_offers", "player", "resource", "road", "stars", "structure", "structure_level", "tech",
    "terrain", "turn", "unit_health", "unit_owner", "unit_type", "unit_veteran",
];

/// Field names of the packed view of a game with `np` seats, in packing order.
pub fn view_fields(np: usize) -> Vec<&'static str> {
    let mut f = VIEW_FIELDS.to_vec();
    if np > 2 {
        f.insert(0, "alive");
    }
    f
}

pub fn view_width(n: usize) -> usize {
    37 * n + 397
}

/// Packed view width for `np` seats: relations take np entries each, plus "alive" when np > 2.
pub fn view_width_np(n: usize, np: usize) -> usize {
    view_width(n) + if np > 2 { 6 * np - 10 } else { 0 }
}

/// Largest board the observation's stack buffers hold (18x18 is the largest map size).
const MAX_TILES: usize = 512;

#[inline(always)]
fn put_sel(dst: &mut [i16], mask: &[bool], v: &[i32], other: i16) {
    for ((d, &m), &x) in dst.iter_mut().zip(mask).zip(v) {
        *d = if m { x as i16 } else { other };
    }
}

#[inline(always)]
fn put_bool(dst: &mut [i16], v: &[bool]) {
    for (d, &x) in dst.iter_mut().zip(v) {
        *d = x as i16;
    }
}

/// visible ? (unit visible ? v : 0) : -1
#[inline(always)]
fn put_unit(dst: &mut [i16], vis: &[bool], uvis: &[bool], v: &[i32]) {
    for (((d, &a), &b), &x) in dst.iter_mut().zip(vis).zip(uvis).zip(v) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
}

/// kernels.observe packed in VIEW_FIELDS order: one tight loop per field over precomputed visibility and
/// ownership masks. The same code is compiled twice and picked at run time: with AVX2 (the i32 -> i16 selects
/// vectorise) when the CPU has it, else for the baseline target. Integer-only, so both give identical bits.
pub fn observe(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2 (checked just above).
            return unsafe { observe_avx2(s, player, g, out) };
        }
    }
    observe_impl(s, player, g, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn observe_avx2(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    observe_impl(s, player, g, out)
}

#[inline(always)]
fn observe_impl(s: &State, player: i32, g: &Geom, out: &mut [i16]) {
    let n = s.n;
    let np = s.np;
    assert!(n <= MAX_TILES && out.len() >= view_width_np(n, np));
    let p = (player - 1) as usize;
    let (t, u, c, pl) = (&s.tiles, &s.units, &s.cities, &s.players);
    let tech = &pl.tech[p];
    let (mut vis, mut uvis, mut unit) = ([false; MAX_TILES], [false; MAX_TILES], [false; MAX_TILES]);
    // the seat's cities and units: their fields are zero everywhere else
    let (mut cities, mut units) = ([0u16; MAX_TILES], [0u16; MAX_TILES]);
    let (mut nc, mut nu) = (0usize, 0usize);
    for i in 0..n {
        let v = t.explored[p][i] || !s.fog;
        vis[i] = v;
        unit[i] = u.owner[i] == player;
        uvis[i] = v && (unit[i] || !u.invisible[i]);
        cities[nc] = i as u16;
        nc += (c.owner[i] == player) as usize;
        units[nu] = i as u16;
        nu += unit[i] as usize;
    }
    let (vis, uvis, unit, cities, units) = (&vis[..n], &uvis[..n], &unit[..n], &cities[..nc], &units[..nu]);
    let mut at = 0usize;
    macro_rules! field {
        ($len:expr) => {{
            let f = &mut out[at..at + $len];
            at += $len;
            f
        }};
    }
    macro_rules! own {
        ($list:expr, $v:expr) => {{
            let f = field!(n);
            f.fill(0);
            for &i in $list {
                f[i as usize] = $v[i as usize] as i16;
            }
        }};
    }
    macro_rules! small {
        ($($v:expr),*) => {{ $( out[at] = $v as i16; at += 1; )* }};
    }
    if np > 2 {
        for k in 0..np {
            small!(s.alive(k));
        }
    }
    put_sel(field!(n), vis, &t.capital, -1);
    let detect = field!(n);
    detect.fill(0);
    for i in 0..n {
        if unit[i] {
            detect[i] = g.adj[i].iter().any(|&j| {
                let j = j as usize;
                u.owner[j] != 0 && u.owner[j] != player && u.invisible[j]
            }) as i16;
        }
    }
    for k in 0..np {
        small!(pl.embassy[k][p]);
    }
    for k in 0..np {
        small!(pl.embassy[p][k]);
    }
    put_bool(field!(n), vis);
    for k in 0..np {
        small!(pl.known[p][k]);
    }
    put_bool(field!(128), &pl.built[p]);
    own!(cities, c.connected);
    own!(cities, c.level);
    own!(cities, c.population);
    own!(cities, c.production);
    own!(cities, c.progress);
    let rewards = field!(n * 9);
    rewards.fill(0);
    for &i in cities {
        let i = i as usize;
        for k in 0..9 {
            rewards[i * 9 + k] = c.rewards[i * 9 + k] as i16;
        }
    }
    own!(cities, c.riot);
    put_bool(field!(128), &pl.monuments[p]);
    own!(units, u.attacked);
    own!(units, u.boosted);
    own!(units, u.frozen);
    own!(units, u.home);
    own!(units, u.invisible);
    own!(units, u.kills);
    own!(units, u.moved);
    own!(units, u.passenger);
    own!(units, u.poison);
    put_sel(field!(n), vis, &t.owner, -1);
    for k in 0..np {
        small!(pl.peace[p][k]);
    }
    for k in 0..np {
        small!(pl.peace_offer[p][k]);
    }
    small!(player);
    let hide2 = !(tech[6] || tech[8] || tech[9]);
    let hide5 = !(tech[20] || tech[23] || tech[24]);
    for ((d, &m), &r) in field!(n).iter_mut().zip(vis).zip(&t.resource) {
        let hidden = (r == 2 && hide2) || (r == 5 && hide5);
        *d = if m { if hidden { 0 } else { r as i16 } } else { -1 };
    }
    for ((d, &m), &r) in field!(n).iter_mut().zip(vis).zip(&t.road) {
        *d = if m { r as i16 } else { -1 };
    }
    small!(pl.stars[p]);
    put_sel(field!(n), vis, &t.structure, -1);
    put_sel(field!(n), vis, &t.structure_level, -1);
    put_bool(field!(128), tech);
    put_sel(field!(n), vis, &t.terrain, -1);
    small!(s.turn);
    // unit_health is float32 in the view; the pack casts it to int16 (toward zero).
    for (((d, &a), &b), &h) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.health) {
        let v: f32 = if a { if b { h } else { 0.0 } } else { -1.0 };
        *d = v as i16;
    }
    put_unit(field!(n), vis, uvis, &u.owner);
    put_unit(field!(n), vis, uvis, &u.kind);
    for (((d, &a), &b), &x) in field!(n).iter_mut().zip(vis).zip(uvis).zip(&u.veteran) {
        *d = if a { if b { x as i16 } else { 0 } } else { -1 };
    }
    debug_assert_eq!(at, view_width_np(n, np));
}
}
mod pathfinding {

use crate::geom::Geom;
use crate::kernels::{bridge_blocks, forest_step, hide_passable, peace_unit, movement_flags, INF, MV_FOREST, MV_LAND, MV_MOUNTAIN, MV_OWN_PORT, MV_PASS, MV_ROAD, MV_WATER, MV_ZOC};
use crate::rules::{Rules, CARRY, CREEP, FLOAT, HIDE, INFILTRATE, SCOUT};
use crate::state::State;

struct Heap {
    ids: Vec<i32>,
    costs: Vec<i32>,
    count: usize,
}

impl Heap {
    fn push(&mut self, index: i32, cost: i32) {
        let mut at = self.count;
        self.ids[at] = index;
        self.costs[at] = cost;
        while at > 0 && self.costs[at] < self.costs[(at - 1) / 2] {
            let parent = (at - 1) / 2;
            self.ids.swap(at, parent);
            self.costs.swap(at, parent);
            at = parent;
        }
        self.count += 1;
    }

    fn pop(&mut self) -> (i32, i32) {
        let (index, cost) = (self.ids[0], self.costs[0]);
        self.count -= 1;
        let count = self.count;
        let (last_id, last_cost) = (self.ids[count], self.costs[count]);
        let mut at = 0usize;
        while 2 * at + 1 < count {
            let left = 2 * at + 1;
            let right = left + 1;
            let child = if right < count && self.costs[right] <= self.costs[left] { right } else { left };
            self.ids[at] = self.ids[child];
            self.costs[at] = self.costs[child];
            at = child;
        }
        self.ids[at] = last_id;
        self.costs[at] = last_cost;
        while at > 0 && self.costs[at] < self.costs[(at - 1) / 2] {
            let p = (at - 1) / 2;
            self.ids.swap(at, p);
            self.costs.swap(at, p);
            at = p;
        }
        (index, cost)
    }
}

/// Reusable buffers for path search and reveal (one per GameBatch, inside core::Scratch).
pub struct PathScratch {
    heap: Heap,
    flags: Vec<u8>,
    dist: Vec<i32>,
    parent: Vec<i32>,
    /// path tiles from the target back to (excluding) the source: path[k] has rank k
    pub path: Vec<usize>,
    pub reveal: Vec<bool>,
}

impl PathScratch {
    pub fn new(n: usize) -> PathScratch {
        PathScratch { heap: Heap { ids: vec![0; 8 * n + 1], costs: vec![INF; 8 * n + 1], count: 0 }, flags: vec![0; n],
                      dist: vec![INF; n], parent: vec![-1; n], path: Vec::with_capacity(n), reveal: vec![false; n] }
    }
}

/// pathfinding.path_details: the reference-ordered shortest path from `source` to `target` (Dijkstra with the
/// reference heap and its tie behaviour, neighbours in kernels' fixed order, movement data's passable/terminal/
/// road/budget rules). Fills `ps.path` (target first, source excluded) and returns whether the target was found;
/// the reference's mask is {source, target} plus the path and its rank is the index in `ps.path` (source: len).
pub fn search(s: &State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) -> bool {
    let n = s.n;
    let u = &s.units;
    let cp = s.current_player;
    let owner = u.owner[source];
    let kind = u.kind[source] as usize;
    let sk = &r.skills[kind];
    let (aquatic, creep, carry) = (sk[FLOAT], sk[CREEP], sk[CARRY]);
    let no_zoc = creep || sk[INFILTRATE] || u.invisible[source];
    let budget = (r.unit_movement[kind] + u.boosted[source] as i32) * 2;
    movement_flags(s, owner, g, &mut ps.flags);
    let flags = &ps.flags;
    let term_bits = (if no_zoc { 0 } else { MV_ZOC }) | MV_MOUNTAIN | (if aquatic { 0 } else { MV_OWN_PORT });
    let terminal = |f: u8| f & term_bits != 0 || (f & (MV_FOREST | MV_ROAD) == MV_FOREST && !creep) || (carry && f & MV_WATER == 0);
    let pass_need = if aquatic { MV_PASS } else { MV_PASS | MV_LAND };
    let hide = sk[HIDE];
    let passable = |i: usize| (flags[i] & pass_need == pass_need && (u.owner[i] == 0 || u.owner[i] == cp || u.invisible[i]))
        || ((hide || peace_unit(s, owner, i)) && (aquatic || flags[i] & MV_LAND != 0) && hide_passable(s, owner, i));
    let (heap, dist, parent) = (&mut ps.heap, &mut ps.dist, &mut ps.parent);
    dist[..n].fill(INF);
    parent[..n].fill(-1);
    heap.ids[0] = source as i32;
    heap.costs[0] = 0;
    heap.count = 1;
    dist[source] = 0;
    let mut found = false;
    while heap.count > 0 && !found {
        let (at, cost) = heap.pop();
        let at = at as usize;
        found = at == target;
        let expand = !found && cost <= dist[at] && (at == source || !terminal(flags[at])) && cost < budget;
        if expand {
            let road_at = flags[at] & MV_ROAD != 0 && !creep;
            for k in 0..8 {
                if !g.nb_valid[at * 8 + k] {
                    continue;
                }
                let dest = g.nb[at * 8 + k] as usize;
                let road_edge = road_at && flags[dest] & MV_ROAD != 0;
                let newcost = forest_step(creep, road_edge, flags[dest], cost + if road_edge { 1 } else { 2 }, budget);
                if passable(dest) && dest != source && newcost < dist[dest] && !bridge_blocks(s, owner, at, dest) {
                    heap.push(dest as i32, newcost);
                    dist[dest] = newcost;
                    parent[dest] = at as i32;
                }
            }
        }
    }
    ps.path.clear();
    if found {
        let mut at = target as i32;
        while at >= 0 && at as usize != source {
            ps.path.push(at as usize);
            at = parent[at as usize];
        }
    }
    found
}

/// pathfinding.reveal_path into `ps.reveal` for the traveled tiles {source} + ps.path[from..] + extra: every tile
/// within 1 (2 from a mountain, or for scouts) of one of them, visiting only the boxes around them.
pub fn reveal_tiles(s: &State, source: usize, from: usize, extra: Option<usize>, r: &Rules, ps: &mut PathScratch) {
    let size = s.size as i32;
    let scout = r.skills[s.units.kind[source] as usize][SCOUT];
    let out = &mut ps.reveal[..s.n];
    out.fill(false);
    for m in std::iter::once(source).chain(ps.path[from..].iter().copied()).chain(extra) {
        let radius = if s.tiles.terrain[m] == 4 || scout { 2 } else { 1 };
        let (mx, my) = ((m % s.size) as i32, (m / s.size) as i32);
        for y in (my - radius).max(0)..=(my + radius).min(size - 1) {
            for x in (mx - radius).max(0)..=(mx + radius).min(size - 1) {
                out[(y * size + x) as usize] = true;
            }
        }
    }
}
}
mod military {

use crate::diplomacy::{hostile, protected};
use crate::economy::discover;
use crate::geom::Geom;
use crate::kernels::{combat, max_health};
use crate::rules::{Rules, CARRY, FLOAT, HIDE, INDEPENDENT, SCOUT, SPLASH, STOMP, SURPRISE};
use crate::state::State;
use crate::structures::capital_index;

pub fn exhaust(s: &mut State, source: usize) {
    s.units.moved[source] = true;
    s.units.attacked[source] = true;
}

/// military.remove: killer=None mirrors killer=-1 (index clipped to 0, no credit).
pub fn remove(s: &mut State, target: usize, r: &Rules, killer: Option<usize>, casualty: bool) {
    let u = &s.units;
    let owner = s.pclip(u.owner[target]);
    let k = killer.unwrap_or(0);
    let kp = s.pclip(u.owner[k]);
    let kind = u.kind[target] as usize;
    let passenger = u.passenger[target];
    let value = r.unit_cost[kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    let loss = -5 * value * (!u.converted[target]) as i32;
    s.players.score[owner] = s.players.score[owner].wrapping_add(loss);
    s.players.casualties[owner] += casualty as i32;
    s.players.kills[kp] += killer.is_some() as i32;
    if s.units.poison[target] && s.tiles.structure[target] == 0 {
        s.tiles.resource[target] = 7;
    }
    // AttackAction.Execute (game version >= 95): no xp for a killer that is a vehicle (Carry: every
    // boat) or Static; the passenger of a boat keeps the xp it had when it embarked
    let xp = killer.is_some() && !(r.skills[u.kind[k] as usize][CARRY] || r.stat[u.kind[k] as usize]);
    s.units.kills[k] += xp as i32;
    s.units.clear(target);
}

pub fn relocate(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, involuntary: bool, reveal: Option<&[bool]>) {
    let (n, size) = (s.n, s.size);
    let owner = s.units.owner[source];
    let kind = s.units.kind[source] as usize;
    let passenger_src = s.units.passenger[source];
    let moved_src = s.units.moved[source];
    let attacked_src = s.units.attacked[source];
    let invisible_src = s.units.invisible[source];
    let converted_src = s.units.converted[source];
    let terrain_src = s.tiles.terrain[source];
    s.units.move_unit(source, target);
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    s.units.prev_x[target] = (source % size) as i32;
    s.units.prev_y[target] = (source / size) as i32;
    let scout = r.skills[kind][SCOUT];
    let own_reveal: Vec<bool>;
    let reveal: &[bool] = match reveal {
        Some(v) => v,
        None => {
            let rs = if terrain_src == 4 || scout { 2 } else { 1 };
            let rt = if s.tiles.terrain[target] == 4 || scout { 2 } else { 1 };
            own_reveal = (0..n).map(|t| g.d(source, t) <= rs || g.d(target, t) <= rt).collect();
            &own_reveal
        }
    };
    discover(s, reveal, owner);
    if r.skills[kind][STOMP] {
        crate::cov::hit(28);
        area_damage(s, target, target, r, g, true);
    }
    let tt = s.tiles.terrain[target];
    let embark = s.tiles.structure[target] == 8 && !r.skills[kind][FLOAT];
    let special_land = (kind == 39 || kind == 40 || kind == 45) && tt > 2;
    let disembark = r.skills[kind][CARRY] && (passenger_src > 0 || special_land) && tt > 2 && !embark;
    let mut new_kind = if embark {
        if kind == 12 { 45 } else if kind == 38 { 39 } else if kind == 37 { 40 } else { 43 }
    } else if disembark {
        passenger_src as usize
    } else {
        kind
    };
    if special_land {
        new_kind = if kind == 39 { 38 } else if kind == 40 { 37 } else { 12 };
    }
    let passenger = if embark && kind != 12 && kind != 38 && kind != 37 { kind as i32 } else if disembark { 0 } else { passenger_src };
    let score_before = r.unit_cost[kind] + if passenger_src > 0 { r.unit_cost[passenger_src as usize] } else { 0 };
    let score_after = r.unit_cost[new_kind] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    if embark || disembark { crate::cov::hit(36); }
    let dash = !involuntary && !moved_src && !attacked_src && r.dash[new_kind];
    s.units.kind[target] = new_kind as i32;
    s.units.passenger[target] = passenger;
    s.units.attacked[target] = if dash { false } else { s.units.attacked[target] || disembark };
    let hidden = r.skills[new_kind][HIDE] && !involuntary;
    s.units.invisible[target] |= hidden;
    s.units.attacked[target] |= hidden && !invisible_src;
    let o = s.pwrap(owner);
    s.players.score[o] = s.players.score[o].wrapping_add(5 * (score_after - score_before) * (!converted_src) as i32);
}

pub fn attack_mask(s: &State, source: usize, r: &Rules, g: &Geom, out: &mut [bool]) {
    out[..s.n].fill(false);
    attack_targets(s, source, r, g, |t| out[t] = true);
}

/// The true entries of attack_mask's row for `source`, visiting only the tiles within the unit's range.
pub fn attack_targets(s: &State, source: usize, r: &Rules, g: &Geom, mut emit: impl FnMut(usize)) {
    let u = &s.units;
    let cp = s.current_player;
    let kind = u.kind[source] as usize;
    let own = u.owner[source] == cp && !u.attacked[source] && u.health[source] > 0.0;
    let mut can = own && (!u.moved[source] || r.dash[kind]) && r.unit_attack[kind] >= 0.0;
    let infil = r.skills[kind][crate::rules::INFILTRATE];
    can = can && (r.unit_attack[kind] > 0.0 || kind == 10 || infil);
    let range = r.unit_range[kind];
    if !can || range < 1 {
        return;
    }
    let stamp = s.turn * s.np as i32 + cp;
    let explored = &s.tiles.explored[s.p()];
    let size = s.size as i32;
    let (sx, sy) = ((source % s.size) as i32, (source / s.size) as i32);
    for y in (sy - range).max(0)..=(sy + range).min(size - 1) {
        for x in (sx - range).max(0)..=(sx + range).min(size - 1) {
            let t = (y * size + x) as usize;
            if t == source || !explored[t] {
                continue;
            }
            let targets = if infil {
                State::foreign(s.cities.owner[t], cp) && !protected(s, cp, s.cities.owner[t])
                    && !(u.owner[t] != 0 && u.owner[t] != s.cities.owner[t]) && s.cities.infiltrated[t] != stamp
            } else {
                State::foreign(u.owner[t], cp) && !u.invisible[t] && !protected(s, cp, u.owner[t])
            };
            if targets {
                debug_assert!(g.d(source, t) > 0 && g.d(source, t) <= range);
                emit(t);
            }
        }
    }
}

#[inline]
fn terrain_bonus(s: &State, r: &Rules, i: usize, tech: &[bool; 128]) -> f32 {
    let (t, u, c) = (&s.tiles, &s.units, &s.cities);
    let terrain = t.terrain[i];
    let city = if c.owner[i] == u.owner[i] && r.fortify[u.kind[i] as usize] {
        if s.reward(i, 1) > 0 { 4.0 } else { 1.5 }
    } else {
        1.0
    };
    let bonus = if terrain == 1 || terrain == 2 {
        if tech[12] { 1.5 } else { 1.0 }
    } else if terrain == 5 {
        if tech[18] { 1.5 } else { 1.0 }
    } else if terrain == 4 {
        if tech[20] { 1.5 } else { 1.0 }
    } else {
        city
    };
    bonus * if u.poison[i] { 0.5 } else { 1.0 }
}

pub fn defense_bonus(s: &State, r: &Rules, i: usize) -> f32 {
    let tech = &s.players.tech[s.pclip(s.units.owner[i])];
    terrain_bonus(s, r, i, tech)
}

pub fn attack(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let size = s.size;
    let cp = s.current_player;
    let kind = s.units.kind[source] as usize;
    let defender = s.units.kind[target] as usize;
    let terrain = s.tiles.terrain[target];
    let tech = s.players.tech[s.pwrap(s.units.owner[target])];
    let bonus = terrain_bonus(s, r, target, &tech);
    let hp_s = max_health(s, r, source);
    let hp_t = max_health(s, r, target);
    let dd = r.unit_defense[defender] * if s.units.poison[target] { 0.7 } else { 1.0 };
    let (damage, retaliation, _) = combat(r.unit_attack[kind], s.units.health[source], hp_s, dd, s.units.health[target], hp_t, bonus);
    let dead = s.units.health[target] - damage <= 0.0;
    // BattleHelpers.CanDefenderRetaliate: the attacker must also be one of the defender's attack
    // options, i.e. stand on a tile the defender's owner has explored
    let seen = s.tiles.explored[s.pclip(s.units.owner[target])][source];
    let retaliate = !r.skills[kind][SURPRISE] && !r.stiff[defender] && g.d(source, target) <= r.unit_range[defender] && seen;
    s.players.attacked[s.p()] = true;
    s.units.health[target] += -damage;
    s.units.boosted[target] = false;
    s.units.prev_x[source] = 2 * (source % size) as i32 - (target % size) as i32;
    s.units.prev_y[source] = 2 * (source / size) as i32 - (target / size) as i32;
    s.units.last_attack[source] = target as i32 + 1;
    if r.skills[kind][SPLASH] {
        crate::cov::hit(27);
        area_damage(s, source, target, r, g, false);
    }
    if dead {
        remove(s, target, r, Some(source), true);
    } else {
        s.units.health[source] += -(if retaliate { retaliation } else { 0.0 });
    }
    let attacker_dead = s.units.health[source] <= 0.0;
    if attacker_dead {
        crate::cov::hit(34);
        remove(s, source, r, Some(target), true);
    }
    let st = s.tiles.structure[target];
    let can_enter = terrain == 3 || terrain == 5 || (terrain == 4 && s.players.tech[s.p()][20])
        || (terrain <= 2 && (r.skills[kind][FLOAT] || st == 48 || (st == 8 && s.tiles.owner[target] == cp)));
    let advance = dead && !attacker_dead && r.unit_range[kind] < 2 && can_enter;
    if advance {
        crate::cov::hit(35);
        relocate(s, source, target, r, g, true, None);
    }
    let at = if advance { target } else { source };
    if !attacker_dead {
        let persist = r.persist[kind];
        let escape = r.escape[kind];
        s.units.attacks_performed[at] += 1;
        s.units.attacked[at] = !(dead && persist);
        s.units.moved[at] = !escape;
        if dead && persist {
            // AttackCommand.ExecuteV104: Persist leaves the unit able to attack only when it has
            // another target from where it now stands
            let mut more = false;
            attack_targets(s, at, r, g, |_| more = true);
            s.units.attacked[at] = !more;
        }
    }
}

pub fn convert(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let old = s.pwrap(s.units.owner[target]);
    let (cap, exists) = capital_index(s, cp);
    let mut fallback = 0usize;
    let mut best = i32::MAX;
    for i in 0..n {
        if s.cities.owner[i] == cp && s.cities.order[i] < best {
            best = s.cities.order[i];
            fallback = i;
        }
    }
    let tk = s.units.kind[target] as usize;
    let home = if r.skills[tk][INDEPENDENT] { 0 } else { (if exists { cap } else { fallback }) as i32 + 1 };
    let passenger = s.units.passenger[target];
    let cost = r.unit_cost[tk] + if passenger > 0 { r.unit_cost[passenger as usize] } else { 0 };
    s.players.score[old] = s.players.score[old].wrapping_sub(5 * cost * (!s.units.converted[target]) as i32);
    s.players.attacked[s.p()] = true;
    s.units.owner[target] = cp;
    s.units.converted[target] = true;
    s.units.home[target] = home;
    s.units.moved[target] = true;
    s.units.attacked[target] = true;
    exhaust(s, source);
    let radius = if s.tiles.terrain[target] == 4 || r.skills[tk][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(target, t) <= radius).collect();
    discover(s, &reveal, cp);
}

/// out[t*128 + kind]
/// The Raft upgrades (Rammership, Scoutship, Bomber): only summonable onto an own Raft.
pub const UPGRADES: u128 = 1 << 41 | 1 << 42 | 1 << 44;

/// Per-call inputs of summon_mask: units per home city and the unlocked, affordable kinds.
pub struct SummonCtx {
    counts: Vec<i32>,
    ok: u128,
}

impl SummonCtx {
    pub fn new(n: usize) -> SummonCtx {
        SummonCtx { counts: vec![0; n], ok: 0 }
    }

    pub fn prepare(&mut self, s: &State, r: &Rules) {
        let n = s.n;
        let u = &s.units;
        let cp = s.current_player;
        let p = s.p();
        self.counts.fill(0);
        for i in 0..n {
            if u.owner[i] == cp && u.home[i] >= 1 && (u.home[i] as usize) <= n {
                self.counts[(u.home[i] - 1) as usize] += 1;
            }
        }
        let mut unlocked = r.unit_static;
        let mut known = crate::rules::bits128(&s.players.tech[p]);
        while known != 0 {
            unlocked |= r.units_by_tech[known.trailing_zeros() as usize];
            known &= known - 1;
        }
        self.ok = unlocked & r.unit_cost_le.le(s.players.stars[p]);
    }
}

/// One row of military.summon_mask as a bitset of unit kinds.
#[inline]
pub fn summon_row(s: &State, ctx: &SummonCtx, t: usize) -> u128 {
    let (u, c) = (&s.units, &s.cities);
    let cp = s.current_player;
    let available = c.owner[t] == cp && ctx.counts[t] <= c.level[t] && u.owner[t] == 0;
    let friendly = s.tiles.owner[t] == cp || (s.tiles.owner[t] > 0 && s.players.peace[s.p()][s.pclip(s.tiles.owner[t])]);
    let upgrade_allowed = u.owner[t] == cp && u.kind[t] == 43 && friendly;
    (if available { ctx.ok & !UPGRADES } else { 0 }) | (if upgrade_allowed { ctx.ok & UPGRADES } else { 0 })
}

pub fn summon_mask(s: &State, r: &Rules, out: &mut [bool]) {
    let mut ctx = SummonCtx::new(s.n);
    ctx.prepare(s, r);
    for t in 0..s.n {
        let bits = summon_row(s, &ctx, t);
        for k in 0..128 {
            out[t * 128 + k] = bits >> k & 1 == 1;
        }
    }
}

pub fn summon(s: &mut State, source: usize, kind: usize, r: &Rules, g: &Geom, costs: bool) {
    let n = s.n;
    let cp = s.current_player;
    s.units.clear(source);
    s.units.owner[source] = cp;
    s.units.kind[source] = kind as i32;
    s.units.health[source] = r.unit_health[kind];
    s.units.home[source] = source as i32 + 1;
    s.units.moved[source] = true;
    s.units.attacked[source] = true;
    s.units.prev_x[source] = -1;
    s.units.prev_y[source] = -1;
    s.units.created[source] = s.turn;
    let p = s.p();
    s.players.stars[p] -= r.unit_cost[kind] * costs as i32;
    s.players.score[p] = s.players.score[p].wrapping_add(5 * r.unit_cost[kind]);
    // TrainAction: ExploreFromTile with UnitDataExtensions.GetSightRange (Scout units see 2 tiles)
    let radius = if s.tiles.terrain[source] == 4 || r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, cp);
}

/// The unit on `i` may be promoted: 3 kills, not yet a veteran, not Static
/// (UnitDataExtensions.CanBePromoted): a boat is judged by its passenger, so a non-Static passenger is promoted at sea.
#[inline]
pub fn promotable(s: &State, r: &Rules, i: usize) -> bool {
    let u = &s.units;
    let judged = if u.passenger[i] > 0 { u.passenger[i] } else { u.kind[i] } as usize;
    !u.veteran[i] && u.kills[i] >= 3 && !r.stat[judged]
}

pub fn promote(s: &mut State, source: usize, r: &Rules) {
    let hp = max_health(s, r, source) + 5.0;
    s.units.veteran[source] = true;
    s.units.health[source] = hp;
}

pub fn disband(s: &mut State, source: usize, r: &Rules) {
    let p = s.p();
    disband_of(s, source, r, p);
}

/// Disband the unit on `source`, refunding half its cost to seat index `p` (its owner).
pub fn disband_of(s: &mut State, source: usize, r: &Rules, p: usize) {
    let kind = s.units.kind[source];
    let passenger = s.units.passenger[source];
    let underlying = if passenger > 0 {
        passenger
    } else if kind == 39 {
        38
    } else if kind == 40 {
        37
    } else if kind == 45 {
        12
    } else {
        kind
    };
    let refund = r.unit_cost[underlying as usize].div_euclid(2);
    remove(s, source, r, None, false);
    s.players.stars[p] += refund;
}

pub fn heal_others(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let cp = s.current_player;
    for i in 0..s.n {
        if g.d(source, i) == 1 && s.units.owner[i] == cp {
            if !s.units.poison[i] {
                let mh = max_health(s, r, i);
                s.units.health[i] = (s.units.health[i] + 4.0).min(mh);
            }
            s.units.poison[i] = false;
        }
    }
    exhaust(s, source);
}

fn sign(v: i32) -> i32 {
    v.signum()
}

/// military.push: reference alternating push directions with its centre tie breaks.
pub fn push(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let (n, size) = (s.n, s.size);
    let sz = size as i32;
    let u = &s.units;
    let owner = u.owner[source];
    if owner <= 0 {
        return;
    }
    let (x, y) = ((source % size) as i32, (source / size) as i32);
    let center = sz / 2;
    let cx = sign(center - x);
    let mut cy = sign(center - y);
    if cx == 0 && cy == 0 {
        cy = 1;
    }
    let (px, py) = (u.prev_x[source], u.prev_y[source]);
    let prev_valid = px >= 0 && py >= 0 && (px != x || py != y);
    let flip = if owner != s.current_player { -1 } else { 1 };
    let (mx, my) = (sign(x - px) * flip, sign(y - py) * flip);
    let la = u.last_attack[source] - 1;
    let kind = u.kind[source] as usize;
    let ranged = r.unit_range[kind] > 1 && u.attacked[source] && !(r.escape[kind] && u.moved[source]);
    let (ax, ay) = (sign(la.rem_euclid(sz) - x) * flip, sign(la.div_euclid(sz) - y) * flip);
    let use_move = prev_valid && (la < 0 || u.moved[source]);
    let dx = if la >= 0 && ranged { ax } else if use_move { mx } else { cx };
    let dy = if la >= 0 && ranged { ay } else if use_move { my } else { cy };
    let dirs: [[i32; 2]; 8] = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]];
    let base = dirs.iter().position(|d| d[0] == dx && d[1] == dy).unwrap_or(0) as i32;
    let offsets = [0i32, -1, 1, -2, 2, -3, 3, 4];
    let mut choices: [[i32; 2]; 8] = [[0; 2]; 8];
    for k in 0..8 {
        let d = dirs[(base + offsets[k]).rem_euclid(8) as usize];
        choices[k] = [d[0] + x, d[1] + y];
    }
    for (a, b) in [(1usize, 2usize), (3, 4), (5, 6)] {
        let dista = (choices[a][0] - center).pow(2) + (choices[a][1] - center).pow(2);
        let distb = (choices[b][0] - center).pow(2) + (choices[b][1] - center).pow(2);
        let va = (choices[a][0] - choices[a][1]).abs();
        let vb = (choices[b][0] - choices[b][1]).abs();
        if distb < dista || (distb == dista && vb < va) {
            choices.swap(a, b);
        }
    }
    crate::cov::hit(29);
    let po = s.pclip(owner);
    let mut target = None;
    let mut first_index = 0usize;
    for (k, ch) in choices.iter().enumerate() {
        let idx = (ch[1] * sz + ch[0]).clamp(0, n as i32 - 1) as usize;
        if k == 0 {
            first_index = idx;
        }
        let in_board = ch[0] >= 0 && ch[0] < sz && ch[1] >= 0 && ch[1] < sz;
        let terrain = s.tiles.terrain[idx];
        let st = s.tiles.structure[idx];
        let mut valid = in_board && u.owner[idx] == 0;
        valid = valid && (terrain == 3 || terrain == 5 || (terrain == 4 && s.players.tech[po][20])
            || (terrain <= 2 && (r.skills[kind][FLOAT] || st == 48 || (st == 8 && s.tiles.owner[idx] == owner))));
        if valid {
            target = Some(idx);
            break;
        }
    }
    let _ = first_index;
    match target {
        Some(t) => relocate(s, source, t, r, g, true, None),
        None => remove(s, source, r, None, true),
    }
}

pub fn superunit(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    push(s, source, r, g);
    summon(s, source, 12, r, g, false);
}

pub fn area_damage(s: &mut State, source: usize, center: usize, r: &Rules, g: &Geom, stomp: bool) {
    let n = s.n;
    let src_owner = s.units.owner[source];
    let src_kind = s.units.kind[source] as usize;
    let hp_s = max_health(s, r, source);
    let ah = s.units.health[source];
    let aa = r.unit_attack[src_kind];
    let mut damage = vec![0f32; n];
    let mut dead = vec![false; n];
    let mut any_hit = false;
    for i in 0..n {
        let u = &s.units;
        let targets = g.d(center, i) == 1 && u.owner[i] != 0 && u.owner[i] != src_owner && !protected(s, src_owner, u.owner[i]);
        if targets {
            let dd = r.unit_defense[u.kind[i] as usize] * if u.poison[i] { 0.7 } else { 1.0 };
            let mut dmg = combat(aa, ah, hp_s, dd, u.health[i], max_health(s, r, i), defense_bonus(s, r, i)).2;
            let _ = stomp;
            dmg = dmg.floor();
            damage[i] = dmg;
            dead[i] = u.health[i] - dmg <= 0.0;
            any_hit |= dmg > 0.0;
        }
    }
    for i in 0..n {
        s.units.health[i] -= damage[i];
    }
    for i in 0..n {
        if dead[i] {
            crate::cov::hit(37);
            remove(s, i, r, Some(source), true);
        }
    }
    if any_hit {
        hostile(s);
    }
}

pub fn upgrade(s: &mut State, source: usize, kind: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let before = max_health(s, r, source);
    let old = s.units.kind[source] as usize;
    s.units.kind[source] = kind as i32;
    let hp = (max_health(s, r, source) - (before - s.units.health[source])).max(1.0);
    s.units.health[source] = hp;
    let p = s.p();
    s.players.stars[p] -= r.unit_cost[kind];
    s.players.score[p] = s.players.score[p].wrapping_add(5 * (r.unit_cost[kind] - r.unit_cost[old]));
    let radius = if r.skills[kind][SCOUT] { 2 } else { 1 };
    let reveal: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    discover(s, &reveal, s.current_player);
}
}
mod infiltration {

use crate::geom::Geom;
use crate::hash::{hash_word, hash_words};
use crate::kernels::{combat, max_health};
use crate::military::{defense_bonus, remove, summon};
use crate::rules::Rules;
use crate::state::State;

/// Spawn tiles for `target`'s city in priority order (eligible only), with the reference's seeded tie noise.
pub fn spawn_order(s: &State, target: usize, g: &Geom) -> Vec<usize> {
    let (n, size) = (s.n, s.size);
    let t = &s.tiles;
    let tech = &s.players.tech[s.p()];
    let mut keyed: Vec<(f32, usize)> = Vec::new();
    for i in 0..n {
        let water = t.terrain[i] <= 2;
        let mut eligible = t.city[i] == target as i32 && s.units.owner[i] == 0;
        eligible &= (t.terrain[i] != 4 || tech[20]) && (t.terrain[i] != 1 || tech[10]) && (t.terrain[i] != 2 || tech[13]);
        if !eligible {
            continue;
        }
        let defense = (t.terrain[i] == 4 && tech[20]) || (t.terrain[i] == 5 && tech[18]) || (water && tech[12]);
        let known_resource = t.resource[i] > 0 && (t.resource[i] != 2 || tech[6] || tech[8] || tech[9])
            && (t.resource[i] != 5 || tech[20] || tech[23] || tech[24]);
        let words = [(i % size) as u32, (i / size) as u32, s.turn as u32];
        let noise = (hash_word(s.initial_seed, hash_words(s.initial_seed, &words)) as f32) * 0.1f32 / 4294967296.0f32;
        let priority = noise + (t.structure[i] == 1) as i32 as f32 * 2.0 - water as i32 as f32 * 1.6 + known_resource as i32 as f32 * 0.5
            + defense as i32 as f32;
        keyed.push((-priority, i));
    }
    // stable argsort on the float key, ties by index
    keyed.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let _ = g;
    keyed.into_iter().map(|(_, i)| i).collect()
}

pub fn execute(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom) {
    let owner = s.current_player;
    let po = s.p();
    let c = &s.cities;
    let mut innate = c.production[target] + (s.tiles.capital[target] == c.owner[target]) as i32;
    innate = (innate + c.progress[target].min(0)).max(0);
    let count = c.level[target].min(5) as usize;
    remove(s, source, r, None, true);
    let enemy = s.units.owner[target] > 0 && s.units.owner[target] != owner;
    let tk = s.units.kind[target] as usize;
    let damage = combat(r.unit_attack[37], r.unit_health[37], r.unit_health[37], r.unit_defense[tk], s.units.health[target],
                        max_health(s, r, target), defense_bonus(s, r, target)).0;
    s.units.health[target] += if enemy { -damage } else { 0.0 };
    let dead = enemy && s.units.health[target] <= 0.0;
    if dead {
        remove(s, target, r, None, true);
    }
    s.players.kills[po] += dead as i32;
    s.players.stars[po] += innate;
    s.cities.infiltrated[target] = s.turn * s.np as i32 + owner;
    s.cities.riot[target] = true;
    let order = spawn_order(s, target, g);
    for (i, &at) in order.iter().enumerate() {
        if i >= count.min(5) {
            break;
        }
        let water = s.tiles.terrain[at] <= 2;
        summon(s, at, if water { 40 } else { 37 }, r, g, false);
        s.units.home[at] = 0;
    }
}
}
mod capture {

use crate::diplomacy::protected;
use crate::discovery::explorer;
use crate::economy::{add_population, claim, discover};
use crate::geom::Geom;
use crate::hash::hash_word;
use crate::military::{exhaust, promote, push, summon};
use crate::rules::{Rules, INDEPENDENT};
use crate::state::State;
use crate::structures::{adjust_building_population, capital_index, connections, population_contribution};

pub fn capture_mask(s: &State, out: &mut [bool]) {
    let u = &s.units;
    let cp = s.current_player;
    for i in 0..s.n {
        let idle = u.owner[i] == cp && !u.moved[i] && !u.attacked[i];
        out[i] = idle && capturable(s, i);
    }
}

/// capture_mask's tile condition (a foreign unprotected village/city, a ruin, or a starfish with Navigation).
#[inline]
pub fn capturable(s: &State, i: usize) -> bool {
    let t = &s.tiles;
    let cp = s.current_player;
    // Star Fishing is a build (GameLogicData.CanBuild), so not on a tile that has an improvement
    (t.structure[i] == 1 && t.owner[i] != cp && !protected(s, cp, t.owner[i])) || t.structure[i] == 2
        || (t.resource[i] == 8 && s.players.tech[s.p()][14] && t.structure[i] == 0)
}

pub fn city(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let before = population_contribution(s, r, g);
    let owner = s.current_player;
    let old = s.tiles.owner[source];
    let enemy = old > 0;
    let old_order = s.cities.order[source];
    let old_territory: Vec<bool> = s.cities.territory[source * n..source * n + n].to_vec();
    let order = (0..n).filter(|&i| s.cities.owner[i] == owner).map(|i| s.cities.order[i]).max().unwrap_or(0).max(0) + 1;
    let area: Vec<bool> = (0..n).map(|t| g.d(source, t) <= 1).collect();
    let membership: Vec<bool> = if enemy { old_territory.clone() } else { area.clone() };
    let transfer = 100 + (s.cities.level[source] - 1) * 50 + s.cities.population[source] * 5 + s.reward(source, 2) * 250;
    let score = if enemy { transfer } else { 100 };
    let old_owners: Vec<i32> = s.cities.owner.clone();
    {
        let c = &mut s.cities;
        c.owner[source] = owner;
        if !enemy {
            c.level[source] = 1;
            c.population[source] = 0;
            c.progress[source] = 0;
            c.production[source] = 1;
            c.border[source] = 1;
            for k in 0..9 {
                c.rewards[source * 9 + k] = 0;
            }
        }
        c.connected[source] = enemy && c.connected[source];
        c.territory[source * n..source * n + n].copy_from_slice(&membership);
        for i in 0..n {
            if enemy && old_owners[i] == old && c.order[i] > old_order {
                c.order[i] -= 1;
            }
        }
        c.order[source] = order;
    }
    let po = (owner - 1) as usize;
    s.players.score[po] = s.players.score[po].wrapping_add(score);
    let oo = s.pclip(old);
    if enemy {
        s.players.score[oo] = s.players.score[oo].wrapping_sub(score);
    }
    let capital_here = s.tiles.capital[source] > 0;
    if enemy && capital_here {
        for k in 0..s.np {
            s.players.embassy[k][oo] = false;
        }
    }
    let eliminated = enemy && !(0..n).any(|i| s.cities.owner[i] == old);
    if eliminated {
        crate::cov::hit(30);
        let oi = (old - 1) as usize;
        s.players.killed_turn[oi] = s.turn.max(1);
        s.players.killer[oi] = owner;
        crate::core::leave(s, oi);
        for i in 0..n {
            if s.units.owner[i] == old {
                // WipePlayerAction: every unit of the wiped player is disbanded, refund included
                crate::military::disband_of(s, i, r, oi);
            }
        }
    }
    // two seats: the reference's end at the first elimination; more: when one seat is left
    s.terminated |= eliminated && (s.np == 2 || s.alive_count() <= 1);
    let claimed: Vec<bool> = (0..n)
        .map(|t| if enemy { membership[t] || t == source } else { area[t] && s.tiles.owner[t] == 0 })
        .collect();
    claim(s, source, &claimed, r);
    adjust_building_population(&before, s, r, g);
    discover(s, if enemy { &old_territory } else { &area }, owner);
    connections(s, owner, g, true);
    if enemy {
        connections(s, old, g, true);
    }
    exhaust(s, source);
    s.units.invisible[source] = false;
    for i in 0..n {
        if s.units.owner[i] != owner && s.units.home[i] == source as i32 + 1 {
            s.units.home[i] = 0;
        }
    }
    let k = s.units.kind[source] as usize;
    s.units.home[source] = if r.skills[k][INDEPENDENT] { 0 } else { source as i32 + 1 };
}

pub fn starfish(s: &mut State, source: usize) {
    exhaust(s, source);
    s.units.invisible[source] = false;
    s.tiles.resource[source] = 0;
    let p = s.p();
    s.players.stars[p] += 8;
}

pub fn ruin(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let p = s.p();
    let cp = s.current_player;
    let tech = s.players.tech[p];
    let mut eligible = [false; 128];
    for k in 0..128 {
        let req = r.free_requires[k];
        let prerequisite = if req >= 0 { tech[req.clamp(0, 127) as usize] } else { r.tech_tier[k] == 1 };
        eligible[k] = r.free_tech[k] && !tech[k] && prerequisite;
    }
    let (cap, has_cap) = capital_index(s, cp);
    let can_explore = s.tiles.terrain[source] != 4
        && (0..n).any(|t| g.d(source, t) <= 2 && t != source && !s.tiles.explored[p][t]);
    let options = [true, eligible.iter().any(|&e| e), has_cap, can_explore, true];
    let seed = hash_word(s.initial_seed, s.seed);
    let total = options.iter().filter(|&&o| o).count() as u32;
    let draw = (seed % total) as i32;
    let mut choice = 0usize;
    let mut seen = 0;
    for (i, &o) in options.iter().enumerate() {
        if o {
            if seen == draw {
                choice = i;
                break;
            }
            seen += 1;
        }
    }
    exhaust(s, source);
    s.units.invisible[source] = false;
    s.seed = seed;
    s.tiles.structure[source] = 0;
    s.tiles.structure_level[source] = 0;
    s.tiles.structure_turn[source] = 0;
    crate::cov::hit(18 + choice);
    match choice {
        0 => s.players.stars[p] += 10,
        1 => {
            let seed = hash_word(s.initial_seed, s.seed);
            let count = eligible.iter().filter(|&&e| e).count() as u32;
            let draw = (seed % count.max(1)) as i32;
            let mut kind = 0usize;
            let mut seen = 0;
            for k in 0..128 {
                if eligible[k] {
                    if seen == draw {
                        kind = k;
                        break;
                    }
                    seen += 1;
                }
            }
            s.seed = seed;
            s.players.tech[p][kind] = true;
            s.players.tech_turn[p][kind] = s.turn;
            s.players.score[p] = s.players.score[p].wrapping_add(100 * r.tech_tier[kind]);
        }
        2 => add_population(s, cap, 3),
        3 => {
            let reveal = explorer(s, source, g);
            discover(s, &reveal, cp);
        }
        _ => {
            let water = s.tiles.terrain[source] <= 2;
            push(s, source, r, g);
            summon(s, source, if water { 44 } else { 11 }, r, g, false);
            promote(s, source, r);
            s.units.passenger[source] = if water { 2 } else { 0 };
            s.units.home[source] = 0;
        }
    }
}
}
mod masks {

use crate::capture;
use crate::core::family_widths;
use crate::diplomacy;
use crate::geom::Geom;
use crate::kernels::{max_health, movement_flags, reach, research_bits, MoveScratch};
use crate::military::{self, SummonCtx};
use crate::rules::Rules;
use crate::state::State;
use crate::structures::{self, BuildCtx};

pub fn mask_width(n: usize) -> usize {
    family_widths(n).iter().sum()
}

/// Bytes per game of the packed masks: ceil(mask_width / 8).
pub fn packed_width(n: usize) -> usize {
    (mask_width(n) + 7) / 8
}

pub fn family_offsets(n: usize) -> [usize; 18] {
    let w = family_widths(n);
    let mut o = [0usize; 18];
    for i in 0..17 {
        o[i + 1] = o[i] + w[i];
    }
    o
}

/// Receives the true entries of one game's flat mask.
pub trait Sink {
    fn set(&mut self, i: usize);
}

/// 0/1 bytes; the buffer must be zeroed.
pub struct Dense<'a>(pub &'a mut [u8]);

impl Sink for Dense<'_> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.0[i] = 1;
    }
}

/// Bits, most significant first (numpy.packbits' default order); the buffer must be zeroed.
pub struct Packed<'a>(pub &'a mut [u8]);

impl Sink for Packed<'_> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.0[i >> 3] |= 0x80 >> (i & 7);
    }
}

/// Buffers reused across legal_masks calls (one per GameBatch).
pub struct MaskScratch {
    pub mv: MoveScratch,
    pub build: BuildCtx,
    pub summon: SummonCtx,
}

impl MaskScratch {
    pub fn new(n: usize) -> MaskScratch {
        MaskScratch { mv: MoveScratch::new(n), build: BuildCtx::new(n), summon: SummonCtx::new(n) }
    }
}

/// Dense masks into `out` (any contents; the first mask_width bytes are overwritten).
pub fn legal_masks(s: &State, r: &Rules, g: &Geom, no_resign: bool, ms: &mut MaskScratch, out: &mut [u8]) {
    out[..mask_width(s.n)].fill(0);
    masks_into(s, r, g, no_resign, ms, &mut Dense(out));
}

/// Forwards to another sink and records every entry in a persistent packed bitset (LegalCache).
pub struct Tee<'a, S: Sink> {
    pub out: &'a mut S,
    pub cache: &'a mut LegalCache,
}

impl<S: Sink> Sink for Tee<'_, S> {
    #[inline]
    fn set(&mut self, i: usize) {
        self.out.set(i);
        self.cache.bits[i >> 3] |= 0x80 >> (i & 7);
        self.cache.list.push(i as u32);
    }
}

/// One game's legal masks from the last tick, kept for the next step's validation (core::step_with): packed bits
/// (masks::Packed layout), the list of set entries (to clear them cheaply) and whether they belong to the game's
/// current state.
pub struct LegalCache {
    pub bits: Vec<u8>,
    list: Vec<u32>,
    pub valid: bool,
}

impl LegalCache {
    pub fn new(n: usize) -> LegalCache {
        LegalCache { bits: vec![0; packed_width(n)], list: Vec::new(), valid: false }
    }
    /// Forget the cached masks (they no longer describe the state).
    pub fn clear(&mut self) {
        for &i in &self.list {
            self.bits[(i >> 3) as usize] = 0;
        }
        self.list.clear();
        self.valid = false;
    }
}

/// Counts the true entries (profiling: the mask computation without its memory traffic).
pub struct Count(pub usize);

impl Sink for Count {
    #[inline]
    fn set(&mut self, _: usize) {
        self.0 += 1;
    }
}

/// Emit every true entry of actions.legal_masks (CpuGames: resign forced False when no_resign).
pub fn masks_into<S: Sink>(s: &State, r: &Rules, g: &Geom, no_resign: bool, ms: &mut MaskScratch, out: &mut S) {
    let active = s.core_supported && !s.terminated && !s.truncated;
    if !active {
        return;
    }
    // reward is gated by `active` only; any reward entry is also State::pending_reward, which gates the rest
    if reward_rows(s, out) {
        return;
    }
    let o = family_offsets(s.n);
    move_rows(s, r, g, &mut ms.mv, out);
    attack_rows(s, r, g, out);
    summon_rows(s, r, &mut ms.summon, out);
    build_rows(s, r, g, &mut ms.build, out);
    let mut bits = research_bits(s, r);
    while bits != 0 {
        out.set(o[4] + bits.trailing_zeros() as usize);
        bits &= bits - 1;
    }
    economy_rows(s, &ms.build.econ_list, out);
    unit_rows(s, r, g, out);
    let other = 3 - s.current_player;
    let capital = diplomacy::opponent_capital(s);
    let abilities = [18, 19, 19, 20, 21, 22];
    let targets = [-1, 0, 1, capital.0 as i32, -1, -1];
    for k in 0..6 {
        if diplomacy::legal_with(s, other, targets[k], abilities[k], capital) {
            out.set(o[14] + k);
        }
    }
    out.set(o[15]);
    // resigning exists only in two-seat games
    if !no_resign && s.np == 2 {
        out.set(o[16]);
    }
}

/// City rewards: the two kinds of the next slot of every own city with one pending. Returns whether any is
/// pending (State::pending_reward).
pub fn reward_rows<S: Sink>(s: &State, out: &mut S) -> bool {
    let o = family_offsets(s.n);
    let (c, cp) = (&s.cities, s.current_player);
    let mut pending = false;
    for i in 0..s.n {
        if c.owner[i] == cp {
            let count = s.reward_count(i);
            if count < c.level[i] - 1 {
                pending = true;
                let (a, b) = match count + 1 { 1 => (3, 4), 2 => (1, 7), 3 => (5, 8), _ => (2, 6) };
                out.set(o[7] + i * 9 + a);
                out.set(o[7] + i * 9 + b);
            }
        }
    }
    pending
}

/// Move rows of the mover's unmoved units (kernels.movement_targets' remaining gate: a supported kind).
pub fn move_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, mv: &mut MoveScratch, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    let movable = |i: usize| u.owner[i] == cp && !u.moved[i] && r.unit_supported[u.kind[i] as usize];
    let Some(first) = (0..n).find(|&i| movable(i)) else {
        return;
    };
    movement_flags(s, cp, g, &mut mv.flags);
    for i in first..n {
        if movable(i) {
            let base = o[0] + i * n;
            reach(s, i, r, g, mv, |j| out.set(base + j));
        }
    }
}

pub fn attack_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    for i in 0..n {
        if u.owner[i] == cp && !u.attacked[i] {
            let base = o[1] + i * n;
            military::attack_targets(s, i, r, g, |j| out.set(base + j));
        }
    }
}

pub fn summon_rows<S: Sink>(s: &State, r: &Rules, ctx: &mut SummonCtx, out: &mut S) {
    let (n, u, cp) = (s.n, &s.units, s.current_player);
    let o = family_offsets(n);
    ctx.prepare(s, r);
    for i in 0..n {
        if s.cities.owner[i] == cp || u.owner[i] == cp {
            let mut bits = military::summon_row(s, ctx, i) & r.supported_bits;
            while bits != 0 {
                out.set(o[2] + i * 128 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
    }
}

/// Build rows; leaves economic_tiles(s) in `ctx.econ` for the economy rows.
pub fn build_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, ctx: &mut BuildCtx, out: &mut S) {
    let o = family_offsets(s.n);
    ctx.prepare(s, r, g);
    // outside the economy only the neutral road and the neutral bridge can be built
    {
        for i in 0..s.n {
            if !ctx.econ[i] && structures::neutral_road(s, r, ctx, i) {
                out.set(o[3] + i * 128 + 3);
            }
            if !ctx.econ[i] && structures::neutral_bridge(s, r, ctx, i) {
                out.set(o[3] + i * 128 + 48);
            }
        }
    }
    for &i in &ctx.econ_list {
        let i = i as usize;
        let mut bits = structures::build_row(s, r, g, ctx, i) & !(1u128 << 37);
        while bits != 0 {
            out.set(o[3] + i * 128 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

/// Harvest, forest and destroy (the economy tiles, as listed by BuildCtx::prepare).
pub fn economy_rows<S: Sink>(s: &State, econ: &[u16], out: &mut S) {
    let o = family_offsets(s.n);
    let t = &s.tiles;
    let pi = s.p();
    let tech = &s.players.tech[pi];
    let stars = s.players.stars[pi];
    for &i in econ {
        let i = i as usize;
        let (resource, structure, terrain) = (t.resource[i], t.structure[i], t.terrain[i]);
        let unlocked = (resource == 1 && tech[15]) || (resource == 3 && tech[10]) || (resource == 6 && tech[6]);
        if structure == 0 && unlocked && stars >= 2 {
            out.set(o[5] + i);
        }
        // forest: clear (3 stars), burn, grow
        if structure == 0 || structure == 3 {
            if terrain == 5 && tech[9] && stars >= 3 {
                out.set(o[6] + i * 3);
            }
            if terrain == 5 && tech[16] {
                out.set(o[6] + i * 3 + 1);
            }
            if terrain == 3 && tech[19] && stars >= 5 {
                out.set(o[6] + i * 3 + 2);
            }
        }
        if structures::destructible(structure) && tech[3] {
            out.set(o[13] + i);
        }
    }
}

/// Capture, recover, promote, disband, heal (the mover's units only).
pub fn unit_rows<S: Sink>(s: &State, r: &Rules, g: &Geom, out: &mut S) {
    let o = family_offsets(s.n);
    let (u, cp) = (&s.units, s.current_player);
    let tech2 = s.players.tech[s.p()][2];
    for i in 0..s.n {
        if u.owner[i] != cp {
            continue;
        }
        let idle = !u.moved[i] && !u.attacked[i];
        if idle && capture::capturable(s, i) {
            out.set(o[8] + i);
        }
        if idle && u.health[i] < max_health(s, r, i) {
            out.set(o[9] + i);
        }
        if crate::military::promotable(s, r, i) {
            out.set(o[10] + i);
        }
        if idle && tech2 {
            out.set(o[11] + i);
        }
        if idle && u.kind[i] == 10 && g.adj[i].iter().any(|&j| {
            let j = j as usize;
            u.owner[j] == cp && u.health[j] < max_health(s, r, j)
        }) {
            out.set(o[12] + i);
        }
    }
}
}
mod core {

use crate::capture;
use crate::cov;
use crate::diplomacy;
use crate::economy;
use crate::geom::Geom;
use crate::infiltration;
use crate::kernels::{max_health, movement_targets, research_cost, research_mask, MoveScratch};
use crate::military;
use crate::prof;
use crate::pathfinding::{reveal_tiles, search, PathScratch};
use crate::rules::{Rules, INFILTRATE};
use crate::state::{State, MAXP};
use crate::structures;

pub const OK: i32 = 0;
pub const INVALID_ACTION: i32 = 1;
pub const UNSUPPORTED_RULE: i32 = 2;
pub const STALE_VERSION: i32 = 3;
pub const EPISODE_DONE: i32 = 4;

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub kind: i32,
    pub source: i32,
    pub target: i32,
    pub parameter: i32,
}

/// Catalog widths of the 17 mask families (gpu/actions.py::decode) for board tiles n.
pub fn family_widths(n: usize) -> [usize; 17] {
    [n * n, n * n, n * 128, n * 128, 128, n, n * 3, n * 9, n, n, n, n, n, n, 6, 1, 1]
}

/// Row widths used by rl/actions.py::flatten_choice (second dim of `shapes`).
pub fn family_row_widths() -> [usize; 17] {
    [0, 0, 128, 128, 128, 1, 3, 9, 1, 1, 1, 1, 1, 1, 6, 1, 1]
}

/// rl.actions.flatten_choice followed by gpu.actions.decode.
pub fn decode(s: &State, family: i32, source: i32, target: i32) -> Action {
    let n = s.n as i64;
    let widths = family_widths(s.n);
    let mut rows = family_row_widths();
    rows[0] = s.n;
    rows[1] = s.n;
    let mut offsets = [0i64; 18];
    for i in 0..17 {
        offsets[i + 1] = offsets[i] + widths[i] as i64;
    }
    let f = family.clamp(0, 16) as usize;
    let flat = offsets[f] + source as i64 * rows[f] as i64 + target as i64;
    let group = (1..=17).filter(|&g| flat >= offsets[g]).count().min(16);
    let at = flat - offsets[group];
    let other = 3 - s.current_player;
    let a = |kind: i32, source: i64, target: i64, parameter: i64| Action {
        kind,
        source: source as i32,
        target: target as i32,
        parameter: parameter as i32,
    };
    match group {
        0 => a(1, at / n, at % n, 0),
        1 => a(2, at / n, at % n, 0),
        2 => a(4, at / 128, -1, at % 128),
        3 => a(6, -1, at / 128, at % 128),
        4 => a(7, -1, -1, at),
        5 => a(5, -1, at, 0),
        6 => a(3, -1, at / 3, at % 3 + 1),
        7 => a(9, -1, at / 9, at % 9),
        8 => a(8, at, -1, 0),
        9 => a(3, at, -1, 7),
        10 => a(3, at, -1, 16),
        11 => a(3, at, -1, 8),
        12 => a(3, at, -1, 9),
        13 => a(3, -1, at, 4),
        14 => {
            let k = (at.clamp(0, 5)) as usize;
            let (capital, _) = diplomacy::opponent_capital(s);
            let targets = [-1i64, 0, 1, capital as i64, -1, -1];
            let abilities = [18i64, 19, 19, 20, 21, 22];
            a(3, other as i64, targets[k], abilities[k])
        }
        15 => a(10, -1, -1, 0),
        _ => a(11, -1, -1, 0),
    }
}

fn meet(s: &mut State) {
    if !s.fog {
        return; // `new` below is false: nothing changes
    }
    let p = s.p();
    let n = s.n;
    // every other seat in order; with two seats exactly the reference's single opponent
    for other in (0..s.np).filter(|&q| q != p) {
        if s.players.known[p][other] {
            continue;
        }
        let sees = (0..n).any(|t| s.tiles.explored[p][t] && s.units.owner[t] == other as i32 + 1 && !s.units.invisible[t]);
        let score = s.players.score[other];
        let mut gift = 1 + 2 * if score > 0 { ((score - 1).div_euclid(1000) + 1).min(5) } else { 0 };
        // game version >= 123 (MeetAction.GetReward / ShouldAddAwayStars): in a two-player game,
        // meeting player 1 pays 2 more ("away stars" for the seat that moves second)
        if s.np == 2 && other == 0 {
            gift += 2;
        }
        if sees {
            s.players.stars[p] += gift;
            s.players.known[p][other] = true;
        }
    }
}

fn research(s: &mut State, a: Action, r: &Rules) {
    let p = s.p();
    let k = a.parameter.clamp(0, 127) as usize;
    let cost = research_cost(s, r, k);
    s.players.stars[p] -= cost;
    s.players.score[p] = s.players.score[p].wrapping_add(r.tech_tier[k] * 100);
    s.players.tech[p][k] = true;
    s.players.tech_turn[p][k] = s.turn;
    meet(s);
}

fn recover(s: &mut State, src: usize, r: &Rules) {
    let cp = s.current_player;
    let territory_owner = s.tiles.owner[src];
    let friendly = territory_owner == cp || (territory_owner > 0 && s.players.peace[s.p()][s.pclip(territory_owner)]);
    let amount = if friendly { 4.0 } else { 2.0 };
    let mh = max_health(s, r, src);
    let u = &mut s.units;
    let hp = if u.poison[src] { u.health[src] } else { (u.health[src] + amount).min(mh) };
    u.health[src] = hp;
    u.poison[src] = false;
    u.moved[src] = true;
    u.attacked[src] = true;
    meet(s);
}

fn move_adjacent(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) {
    let cp = s.current_player;
    let found = search(s, source, target, r, g, ps);
    // The reference's path mask is {source, target} + ps.path with rank = index in ps.path (source: len, target: 0).
    // A hidden enemy on the path stops the move on the stoppable path tile just before the first one met from the
    // source side (`contact`, the hidden tile of highest rank); ranks along a path are distinct, so the reference's
    // argmax/argmin pick the same tiles.
    let hidden = |i: usize| s.units.invisible[i] && s.units.owner[i] != cp;
    let contact = if found { ps.path.iter().rposition(|&i| hidden(i)) } else { None };
    let (stop, traveled_from) = match contact {
        Some(c) => {
            // tiles of rank > c: path[c+1..] and the source; the first stoppable one by increasing rank
            let stop = ps.path[c + 1..].iter().copied().find(|&i| s.units.owner[i] == 0).unwrap_or(source);
            (stop, c + 1)
        }
        None => (target, 0),
    };
    // traveled: the source plus path tiles of rank >= traveled_from; without a path the mask is {source, target}
    let extra = if !found && target != source { Some(target) } else { None };
    reveal_tiles(s, source, traveled_from, extra, r, ps);
    if let Some(c) = contact {
        cov::hit(33);
        let tile = ps.path[c];
        s.units.invisible[tile] = false;
    }
    if stop != source {
        military::relocate(s, source, stop, r, g, false, Some(&ps.reveal));
    }
    meet(s);
}

fn end_turn(s: &mut State, r: &Rules, g: &Geom) {
    let n = s.n;
    let owner = s.current_player;
    let old = s.p();
    for i in 0..n {
        let u = &s.units;
        let idle = u.owner[i] == owner && !u.moved[i] && !u.attacked[i];
        if idle {
            if !u.poison[i] {
                let to = s.tiles.owner[i];
                let friendly = to == owner || (to > 0 && s.players.peace[old][s.pclip(to)]);
                let amount = if friendly { 4.0 } else { 2.0 };
                let mh = max_health(s, r, i);
                s.units.health[i] = (s.units.health[i] + amount).min(mh);
            }
            s.units.poison[i] = false;
        }
    }
    let p = &s.players;
    let (new_owner, turn, native) = if s.np == 2 {
        let dead = [p.killed_turn[0] > 0 || p.resigned_turn[0] > 0, p.killed_turn[1] > 0 || p.resigned_turn[1] > 0];
        let new_owner = if dead[(2 - owner) as usize] { owner } else { 3 - owner };
        (new_owner, s.turn + (new_owner == 1) as i32, dead[0] || dead[1])
    } else {
        // the next seat still in the game, in seat order; a round ends when the order wraps
        let np = s.np as i32;
        let new_owner = (1..=np).map(|k| (owner - 1 + k) % np + 1).find(|&o| s.alive((o - 1) as usize)).unwrap_or(owner);
        (new_owner, s.turn + (new_owner <= owner) as i32, s.alive_count() <= 1)
    };
    let pacifist = if p.attacked[old] { 0 } else { p.pacifist[old] + p.tech[old][21] as i32 };
    s.players.pacifist[old] = pacifist;
    s.players.attacked[old] = false;
    s.current_player = new_owner;
    s.turn = turn;
    let limit = turn > s.reference_max_turns;
    if limit || native {
        if limit && !native { cov::hit(31); }
        s.truncated = limit && !native;
        s.terminated = native;
        return;
    }
    let frozen: Vec<bool> = s.units.frozen.clone();
    diplomacy::start_turn(s, new_owner);
    // StartTurnAction explores nothing: tiles are explored when units move, train, upgrade or convert and when
    // borders grow
    meet(s);
    structures::start_turn(s, new_owner);
    let mut income = structures::income(s, new_owner, r, g);
    if !(turn > 0 && income > 0) {
        income = 0;
    }
    for i in 0..n {
        if s.cities.riot[i] && s.cities.owner[i] == new_owner {
            s.cities.riot[i] = false;
        }
    }
    let no = (new_owner - 1) as usize;
    s.players.stars[no] += income;
    for i in 0..n {
        if s.units.owner[i] == new_owner {
            s.units.moved[i] = frozen[i];
            s.units.attacked[i] = frozen[i];
            if !frozen[i] {
                s.units.attacks_performed[i] = 0;
            }
            s.units.frozen[i] = false;
        }
    }
}

fn resign(s: &mut State) {
    let p = s.p();
    s.players.resigned_turn[p] = s.turn.max(1);
    if s.np == 2 {
        s.terminated = true;
        s.players.killer[p] = 3 - s.current_player;
    } else {
        // the seat leaves (its pieces stay as obstacles; the turn order skips it); the game ends at one seat left
        leave(s, p);
        s.terminated = s.alive_count() <= 1;
    }
}

/// Record that seat `k` has just left a multi-seat game (killed or resigned): its place in the leaving order.
pub fn leave(s: &mut State, k: usize) {
    if s.np > 2 && s.players.out_order[k] == 0 {
        s.players.out_order[k] = s.players.out_order.iter().filter(|&&o| o > 0).count() as i32 + 1;
    }
}

/// Reward of finishing place `place` (0 = first) among `np` seats: evenly spaced from +1 down to -1, zero-sum.
#[inline]
pub fn place_reward(place: usize, np: usize) -> f32 {
    1.0 - 2.0 * place as f32 / (np - 1) as f32
}

/// Seat-indexed final rewards (rows >= np are 0). Two seats: the reference's win +1 / loss -1, 0 unless terminated.
/// More seats: rank rewards (place_reward), set at termination and at truncation. Seats that left take the
/// places from last upward in the order they left; the seats still in share the top places equally (a seat
/// still in at the round cap ties with every other one still in).
pub fn terminal_rewards(s: &State) -> [f32; MAXP] {
    let p = &s.players;
    let mut out = [0f32; MAXP];
    if s.np == 2 {
        let dead = [p.killed_turn[0] > 0 || p.resigned_turn[0] > 0, p.killed_turn[1] > 0 || p.resigned_turn[1] > 0];
        if s.terminated {
            for k in 0..2 {
                let winner = !dead[k] && dead[1 - k];
                out[k] = if winner { 1.0 } else if dead[k] { -1.0 } else { 0.0 };
            }
        }
        return out;
    }
    if !(s.terminated || s.truncated) {
        return out;
    }
    let np = s.np;
    let alive = s.alive_count();
    let shared = (0..alive).map(|k| place_reward(k, np)).sum::<f32>() / alive.max(1) as f32;
    for k in 0..np {
        out[k] = if s.alive(k) { shared } else { place_reward(np - p.out_order[k].clamp(1, np as i32) as usize, np) };
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Research, Recover, Move, End, Attack, Summon, Harvest, Forest, Promote, Disband, Reward, Build, Destroy, Cap,
    Heal, Diplomatic, Resign, Unsupported,
}

fn classify(a: Action) -> Kind {
    match (a.kind, a.parameter) {
        (7, _) => Kind::Research,
        (3, 7) => Kind::Recover,
        (1, _) => Kind::Move,
        (10, _) => Kind::End,
        (2, _) => Kind::Attack,
        (4, _) => Kind::Summon,
        (5, _) => Kind::Harvest,
        (3, 1..=3) => Kind::Forest,
        (3, 16) => Kind::Promote,
        (3, 8) => Kind::Disband,
        (9, _) => Kind::Reward,
        (6, _) => Kind::Build,
        (3, 4) => Kind::Destroy,
        (8, _) => Kind::Cap,
        (3, 9) => Kind::Heal,
        (3, 18..=22) => Kind::Diplomatic,
        (11, _) => Kind::Resign,
        _ => Kind::Unsupported,
    }
}

/// Scratch buffers reused across steps.
pub struct Scratch {
    pub row: Vec<bool>,
    pub mv: MoveScratch,
    pub build: structures::BuildCtx,
    pub summon: military::SummonCtx,
    pub path: PathScratch,
}

impl Scratch {
    pub fn new(n: usize) -> Scratch {
        Scratch { row: vec![false; n * 9], mv: MoveScratch::new(n), build: structures::BuildCtx::new(n),
                  summon: military::SummonCtx::new(n), path: PathScratch::new(n) }
    }
}

/// core.step: apply one atomic action; rejection leaves the state untouched. Returns the status code.
pub fn step(s: &mut State, a: Action, expected_version: i32, r: &Rules, g: &Geom, max_actions: i32, sc: &mut Scratch) -> i32 {
    step_with(s, a, expected_version, r, g, max_actions, sc, None)
}

/// step, given optionally the packed legal masks (masks::Packed layout) computed for exactly this state: Move,
/// Build and Summon are then validated by one mask lookup. The status is the same either way: where a mask entry
/// and step's own predicate can differ (a pending city reward, the kind-37 build, an unsupported unit kind, a
/// core_supported=False state) the action is unsupported regardless of validity.
#[allow(clippy::too_many_arguments)]
pub fn step_with(s: &mut State, a: Action, expected_version: i32, r: &Rules, g: &Geom, max_actions: i32, sc: &mut Scratch,
                 legal: Option<&[u8]>) -> i32 {
    let n = s.n as i32;
    let legal = if s.core_supported { legal } else { None };
    let o = crate::masks::family_offsets(s.n);
    let bit = |b: &[u8], i: usize| b[i >> 3] >> (7 - (i & 7)) & 1 == 1;
    let src = a.source.clamp(0, n - 1) as usize;
    let dst = a.target.clamp(0, n - 1) as usize;
    let param = a.parameter.clamp(0, 127) as usize;
    let source_valid = a.source >= 0 && a.source < n;
    let target_valid = a.target >= 0 && a.target < n;
    let param_valid = a.parameter >= 0 && a.parameter < 128;
    let cp = s.current_player;
    let kind = classify(a);
    let own = source_valid && s.units.owner[src] == cp;
    let idle = own && !s.units.moved[src] && !s.units.attacked[src];
    let pending = s.pending_reward();
    let reward = kind == Kind::Reward;
    let reward_implemented = matches!(a.parameter, 1..=8);
    let mut supported = s.core_supported && kind != Kind::Unsupported && (!pending || reward || kind == Kind::Resign)
        && (!reward || reward_implemented);
    if kind == Kind::Summon {
        supported &= r.unit_supported[param];
    }
    if kind == Kind::Build {
        supported &= param != 37;
    }
    let valid = prof::time(prof::VALIDATE + kind as usize, || match kind {
        Kind::End | Kind::Resign => true,
        Kind::Research => {
            let mut m = [false; 128];
            research_mask(s, r, &mut m);
            param_valid && m[param]
        }
        Kind::Recover => idle && s.units.health[src] < max_health(s, r, src),
        Kind::Move => source_valid && target_valid && match legal {
            Some(b) => bit(b, o[0] + src * s.n + dst),
            None => {
                movement_targets(s, src, r, g, &mut sc.mv, &mut sc.row);
                sc.row[dst]
            }
        },
        Kind::Attack => {
            military::attack_mask(s, src, r, g, &mut sc.row);
            source_valid && target_valid && sc.row[dst]
        }
        Kind::Summon => source_valid && param_valid && match legal {
            Some(b) => bit(b, o[2] + src * 128 + param),
            None => {
                sc.summon.prepare(s, r);
                military::summon_row(s, &sc.summon, src) >> param & 1 == 1
            }
        },
        Kind::Harvest => {
            economy::harvest_mask(s, &mut sc.row);
            target_valid && sc.row[dst]
        }
        Kind::Forest => {
            economy::forest_masks(s, &mut sc.row);
            target_valid && sc.row[dst * 3 + (a.parameter - 1).clamp(0, 2) as usize]
        }
        Kind::Promote => own && military::promotable(s, r, src),
        Kind::Disband => idle && s.players.tech[s.p()][2],
        Kind::Build => target_valid && param_valid && match legal {
            Some(b) => bit(b, o[3] + dst * 128 + param),
            None => structures::build_legal(s, r, g, &mut sc.build, dst, param),
        },
        Kind::Destroy => {
            structures::destroy_mask(s, &mut sc.row);
            target_valid && sc.row[dst]
        }
        Kind::Heal => {
            idle && s.units.kind[src] == 10
                && (0..s.n).any(|i| g.d(src, i) == 1 && s.units.owner[i] == cp && s.units.health[i] < max_health(s, r, i))
        }
        Kind::Cap => {
            capture::capture_mask(s, &mut sc.row);
            source_valid && sc.row[src]
        }
        Kind::Diplomatic => diplomacy::legal(s, a.source, a.target, a.parameter),
        Kind::Reward => {
            economy::reward_mask(s, &mut sc.row);
            target_valid && a.parameter >= 0 && a.parameter < 9 && sc.row[dst * 9 + a.parameter.clamp(0, 8) as usize]
        }
        Kind::Unsupported => false,
    });
    let mut status = if !supported { UNSUPPORTED_RULE } else if valid { OK } else { INVALID_ACTION };
    if expected_version != s.state_version {
        status = STALE_VERSION;
    }
    if s.terminated || s.truncated {
        status = EPISODE_DONE;
    }
    if status != OK {
        return status;
    }
    if !s.monuments_settled {
        prof::time(prof::STEP_PRE, || structures::remember_monuments(s, r, g));
    }
    prof::time(prof::EXECUTE + kind as usize, || match kind {
        Kind::Research => { cov::hit(0); research(s, a, r) }
        Kind::Recover => { cov::hit(1); recover(s, src, r) }
        Kind::Move => { cov::hit(2); move_adjacent(s, src, dst, r, g, &mut sc.path) }
        Kind::End => { cov::hit(3); end_turn(s, r, g) }
        Kind::Attack => {
            let uk = s.units.kind[src] as usize;
            if r.skills[uk][INFILTRATE] {
                cov::hit(6);
                infiltration::execute(s, src, dst, r, g);
            } else if uk == 10 {
                cov::hit(5);
                military::convert(s, src, dst, r, g);
            } else {
                cov::hit(4);
                military::attack(s, src, dst, r, g);
            }
        }
        Kind::Summon => {
            if s.units.kind[src] == 43 {
                cov::hit(8);
                military::upgrade(s, src, param, r, g);
            } else {
                cov::hit(7);
                military::summon(s, src, param, r, g, true);
            }
        }
        Kind::Harvest => { cov::hit(9); economy::harvest(s, dst) }
        Kind::Forest => { cov::hit(10); economy::forest(s, dst, a.parameter) }
        Kind::Promote => { cov::hit(11); military::promote(s, src, r) }
        Kind::Disband => { cov::hit(12); military::disband(s, src, r) }
        Kind::Reward => { cov::hit(13); if a.parameter == 6 { cov::hit(38); } if a.parameter == 4 { cov::hit(39); } economy::reward(s, dst, a.parameter, r, g) }
        Kind::Build => { cov::hit(14); structures::build(s, dst, param, r, g) }
        Kind::Destroy => { cov::hit(15); structures::destroy(s, dst, r, g) }
        Kind::Cap => {
            let st = s.tiles.structure[src];
            if st == 1 {
                cov::hit(16);
                if s.tiles.owner[src] > 0 { cov::hit(17); }
                capture::city(s, src, r, g);
            } else if st == 2 {
                capture::ruin(s, src, r, g);
            } else {
                cov::hit(23);
                capture::starfish(s, src);
            }
        }
        Kind::Heal => { cov::hit(24); military::heal_others(s, src, r, g) }
        Kind::Diplomatic => { cov::hit(25); diplomacy::execute(s, a.target, a.parameter, r, g) }
        Kind::Resign => { cov::hit(26); resign(s) }
        Kind::Unsupported => unreachable!(),
    });
    prof::time(prof::STEP_POST, || {
        if a.kind == 2 {
            diplomacy::hostile(s);
        }
        meet(s);
        diplomacy::capital_vision(s);
        structures::remember_monuments(s, r, g);
    });
    s.monuments_settled = true;
    let count = s.actions + 1;
    s.state_version += 1;
    s.actions = count;
    if count >= max_actions && !s.terminated && !s.truncated { cov::hit(32); }
    s.truncated = (s.truncated || count >= max_actions) && !s.terminated;
    OK
}
}
const RULES_JSON: &str = r###"{
  "engine_revision": "a9f90ab54607f5633556d02422d7457f548cc58f",
  "structures": [
    {
      "adjacent": [],
      "cost": null,
      "id": 0,
      "name": "None",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 1,
      "name": "Village",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 2,
      "name": "Ruin",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 3,
      "id": 3,
      "name": "Road",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3,
        5,
        7,
        8
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 5,
      "name": "Farm",
      "per_city": false,
      "per_tribe": false,
      "population": 2,
      "resource": 2,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [
        5
      ],
      "cost": 5,
      "id": 6,
      "name": "Windmill",
      "per_city": true,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 7,
      "id": 8,
      "name": "Port",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 3,
      "id": 12,
      "name": "LumberHut",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        5
      ],
      "tribe": null
    },
    {
      "adjacent": [
        12
      ],
      "cost": 5,
      "id": 13,
      "name": "Sawmill",
      "per_city": true,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 17,
      "name": "Temple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 15,
      "id": 18,
      "name": "ForestTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        5
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 19,
      "name": "WaterTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1,
        2
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 20,
      "name": "MountainTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        4
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 21,
      "name": "Mine",
      "per_city": false,
      "per_tribe": false,
      "population": 2,
      "resource": 5,
      "score": 0,
      "stars": 0,
      "terrain": [
        4
      ],
      "tribe": null
    },
    {
      "adjacent": [
        21
      ],
      "cost": 5,
      "id": 22,
      "name": "Forge",
      "per_city": true,
      "per_tribe": false,
      "population": 2,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3,
        5
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 23,
      "name": "AltarOfPeace",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 24,
      "name": "TowerOfWisdom",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 25,
      "name": "GrandBazaar",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 26,
      "name": "EmperorsTomb",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 27,
      "name": "GateOfPower",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 28,
      "name": "ParkOfFortune",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 29,
      "name": "EyeOfGod",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        6
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 32,
      "name": "Sanctuary",
      "per_city": true,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 1,
      "terrain": [
        3,
        4,
        5
      ],
      "tribe": 5
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 33,
      "name": "Outpost",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        6
      ],
      "tribe": 16
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 34,
      "name": "IceBank",
      "per_city": false,
      "per_tribe": true,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        3,
        6
      ],
      "tribe": 16
    },
    {
      "adjacent": [],
      "cost": 20,
      "id": 35,
      "name": "IceTemple",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        6
      ],
      "tribe": 16
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 37,
      "name": "Fungi",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": 7,
      "score": 0,
      "stars": 0,
      "terrain": [
        3
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 38,
      "name": "Algae",
      "per_city": false,
      "per_tribe": false,
      "population": 1,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1,
        2
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 39,
      "name": "Mycelium",
      "per_city": true,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1,
        2,
        3,
        4,
        5
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 41,
      "name": "Clathrus",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 1,
      "terrain": [
        1,
        2
      ],
      "tribe": 17
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 47,
      "name": "Lighthouse",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 48,
      "name": "Bridge",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [
        1
      ],
      "tribe": null
    },
    {
      "adjacent": [
        6,
        13,
        22
      ],
      "cost": 5,
      "id": 50,
      "name": "Market",
      "per_city": true,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 1,
      "terrain": [
        3
      ],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": 5,
      "id": 70,
      "name": "Embassy",
      "per_city": false,
      "per_tribe": false,
      "population": 0,
      "resource": null,
      "score": 0,
      "stars": 0,
      "terrain": [],
      "tribe": null
    },
    {
      "adjacent": [],
      "cost": null,
      "id": 123,
      "name": "ChurchOfConverts",
      "per_city": false,
      "per_tribe": false,
      "population": 3,
      "resource": null,
      "score": 400,
      "stars": 0,
      "terrain": [
        1,
        3,
        5
      ],
      "tribe": null
    }
  ],
  "technologies": [
    {
      "id": -1,
      "name": "BeyondComprehension",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 0,
      "name": "Basic",
      "next": [
        1,
        6,
        20,
        10,
        15
      ],
      "requires": null,
      "structures": [],
      "tier": 0,
      "units": [
        2
      ]
    },
    {
      "id": 1,
      "name": "Riding",
      "next": [
        2,
        4
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": [
        3
      ]
    },
    {
      "id": 2,
      "name": "FreeSpirit",
      "next": [
        3
      ],
      "requires": 1,
      "structures": [
        17
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 3,
      "name": "Chivalry",
      "next": [],
      "requires": 2,
      "structures": [],
      "tier": 3,
      "units": [
        4
      ]
    },
    {
      "id": 4,
      "name": "Roads",
      "next": [
        5
      ],
      "requires": 1,
      "structures": [
        3,
        48,
        39
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 5,
      "name": "Trade",
      "next": [],
      "requires": 4,
      "structures": [
        50,
        41
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 6,
      "name": "Organization",
      "next": [
        7,
        8
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 7,
      "name": "Strategy",
      "next": [
        38
      ],
      "requires": 6,
      "structures": [],
      "tier": 2,
      "units": [
        5
      ]
    },
    {
      "id": 8,
      "name": "Farming",
      "next": [
        9
      ],
      "requires": 6,
      "structures": [
        5,
        37
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 9,
      "name": "Construction",
      "next": [],
      "requires": 8,
      "structures": [
        6
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 10,
      "name": "Fishing",
      "next": [
        13,
        44
      ],
      "requires": null,
      "structures": [
        8
      ],
      "tier": 1,
      "units": [
        43
      ]
    },
    {
      "id": 12,
      "name": "Aquatism",
      "next": [],
      "requires": 44,
      "structures": [
        19
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 13,
      "name": "Sailing",
      "next": [
        14
      ],
      "requires": 10,
      "structures": [],
      "tier": 2,
      "units": [
        42
      ]
    },
    {
      "id": 14,
      "name": "Navigation",
      "next": [],
      "requires": 44,
      "structures": [],
      "tier": 3,
      "units": [
        41
      ]
    },
    {
      "id": 15,
      "name": "Hunting",
      "next": [
        18,
        16
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 16,
      "name": "Forestry",
      "next": [
        17
      ],
      "requires": 15,
      "structures": [
        12,
        32
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 17,
      "name": "Mathematics",
      "next": [],
      "requires": 16,
      "structures": [
        13
      ],
      "tier": 3,
      "units": [
        8
      ]
    },
    {
      "id": 18,
      "name": "Archery",
      "next": [
        19
      ],
      "requires": 15,
      "structures": [],
      "tier": 2,
      "units": [
        9
      ]
    },
    {
      "id": 19,
      "name": "Spiritualism",
      "next": [],
      "requires": 18,
      "structures": [
        18
      ],
      "tier": 3,
      "units": []
    },
    {
      "id": 20,
      "name": "Climbing",
      "next": [
        23,
        21
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 21,
      "name": "Meditation",
      "next": [
        22
      ],
      "requires": 20,
      "structures": [
        20
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 22,
      "name": "Philosophy",
      "next": [],
      "requires": 21,
      "structures": [],
      "tier": 3,
      "units": [
        10
      ]
    },
    {
      "id": 23,
      "name": "Mining",
      "next": [
        24
      ],
      "requires": 20,
      "structures": [
        21
      ],
      "tier": 2,
      "units": []
    },
    {
      "id": 24,
      "name": "Smithery",
      "next": [],
      "requires": 23,
      "structures": [
        22
      ],
      "tier": 3,
      "units": [
        11
      ]
    },
    {
      "id": 25,
      "name": "FreeDiving",
      "next": [
        3
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 26,
      "name": "Spearing",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 28,
      "name": "ForestMagic",
      "next": [
        18,
        16
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 30,
      "name": "Frostwork",
      "next": [
        32
      ],
      "requires": null,
      "structures": [
        33
      ],
      "tier": 1,
      "units": []
    },
    {
      "id": 31,
      "name": "PolarWarfare",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 32,
      "name": "Polarism",
      "next": [],
      "requires": null,
      "structures": [
        35
      ],
      "tier": 1,
      "units": []
    },
    {
      "id": 35,
      "name": "ShockTactics",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 36,
      "name": "Recycling",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 37,
      "name": "Hydrology",
      "next": [
        14
      ],
      "requires": null,
      "structures": [
        38
      ],
      "tier": 1,
      "units": []
    },
    {
      "id": 38,
      "name": "Diplomacy",
      "next": [],
      "requires": 7,
      "structures": [
        70
      ],
      "tier": 3,
      "units": [
        38
      ]
    },
    {
      "id": 41,
      "name": "Sledding",
      "next": [
        31
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 42,
      "name": "IceFishing",
      "next": [
        32
      ],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 44,
      "name": "Ramming",
      "next": [
        12
      ],
      "requires": 10,
      "structures": [],
      "tier": 2,
      "units": [
        44
      ]
    },
    {
      "id": 48,
      "name": "Waterways",
      "next": [],
      "requires": 1,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 49,
      "name": "Oceantology",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 120,
      "name": "Synergy",
      "next": [],
      "requires": null,
      "structures": [],
      "tier": 1,
      "units": []
    },
    {
      "id": 121,
      "name": "Rituals",
      "next": [
        22
      ],
      "requires": 20,
      "structures": [],
      "tier": 2,
      "units": []
    }
  ],
  "units": [
    {
      "attack": -1.0,
      "cost": -1,
      "defense": 0.0,
      "health": 1.0,
      "id": 0,
      "movement": 1,
      "name": "None",
      "range": 1,
      "skills": []
    },
    {
      "attack": 2.0,
      "cost": 2,
      "defense": 2.0,
      "health": 10.0,
      "id": 2,
      "movement": 1,
      "name": "Warrior",
      "range": 1,
      "skills": [
        "Dash",
        "Fortify"
      ]
    },
    {
      "attack": 2.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 3,
      "movement": 2,
      "name": "Rider",
      "range": 1,
      "skills": [
        "Dash",
        "Escape",
        "Fortify"
      ]
    },
    {
      "attack": 3.5,
      "cost": 8,
      "defense": 1.0,
      "health": 10.0,
      "id": 4,
      "movement": 3,
      "name": "Knight",
      "range": 1,
      "skills": [
        "Dash",
        "Fortify",
        "Persist"
      ]
    },
    {
      "attack": 1.0,
      "cost": 3,
      "defense": 3.0,
      "health": 15.0,
      "id": 5,
      "movement": 1,
      "name": "Defender",
      "range": 1,
      "skills": [
        "Fortify"
      ]
    },
    {
      "attack": 4.0,
      "cost": 8,
      "defense": 0.0,
      "health": 10.0,
      "id": 8,
      "movement": 1,
      "name": "Catapult",
      "range": 3,
      "skills": [
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 9,
      "movement": 1,
      "name": "Archer",
      "range": 2,
      "skills": [
        "Dash",
        "Fortify"
      ]
    },
    {
      "attack": 0.0,
      "cost": 5,
      "defense": 1.0,
      "health": 10.0,
      "id": 10,
      "movement": 1,
      "name": "MindBender",
      "range": 1,
      "skills": [
        "Convert",
        "HealOthers",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 15.0,
      "id": 11,
      "movement": 1,
      "name": "Swordsman",
      "range": 1,
      "skills": [
        "Dash"
      ]
    },
    {
      "attack": 5.0,
      "cost": 10,
      "defense": 4.0,
      "health": 40.0,
      "id": 12,
      "movement": 1,
      "name": "Giant",
      "range": 1,
      "skills": [
        "Static"
      ]
    },
    {
      "attack": 3.0,
      "cost": 2,
      "defense": 1.0,
      "health": 15.0,
      "id": 15,
      "movement": 1,
      "name": "Polytaur",
      "range": 1,
      "skills": [
        "Dash",
        "Fortify",
        "Independent",
        "Static"
      ]
    },
    {
      "attack": -1.0,
      "cost": 10,
      "defense": 2.0,
      "health": 10.0,
      "id": 17,
      "movement": 1,
      "name": "DragonEgg",
      "range": 1,
      "skills": [
        "Fortify",
        "Grow",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 10,
      "defense": 3.0,
      "health": 15.0,
      "id": 18,
      "movement": 2,
      "name": "BabyDragon",
      "range": 1,
      "skills": [
        "Dash",
        "Escape",
        "Fly",
        "Grow",
        "Scout",
        "Static"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 3.0,
      "health": 20.0,
      "id": 19,
      "movement": 3,
      "name": "FireDragon",
      "range": 2,
      "skills": [
        "Dash",
        "Fly",
        "Scout",
        "Splash",
        "Static"
      ]
    },
    {
      "attack": 2.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 20,
      "movement": 2,
      "name": "Amphibian",
      "range": 1,
      "skills": [
        "Amphibious",
        "Dash",
        "Escape",
        "Fortify"
      ]
    },
    {
      "attack": 2.5,
      "cost": 8,
      "defense": 1.0,
      "health": 10.0,
      "id": 21,
      "movement": 2,
      "name": "Tridention",
      "range": 2,
      "skills": [
        "Amphibious",
        "Dash",
        "Persist"
      ]
    },
    {
      "attack": 0.0,
      "cost": 5,
      "defense": 1.0,
      "health": 10.0,
      "id": 22,
      "movement": 1,
      "name": "Mooni",
      "range": 1,
      "skills": [
        "AutoFreeze",
        "Skate",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 2.0,
      "health": 15.0,
      "id": 23,
      "movement": 2,
      "name": "BattleSled",
      "range": 1,
      "skills": [
        "Dash",
        "Escape",
        "Skate"
      ]
    },
    {
      "attack": 4.0,
      "cost": 15,
      "defense": 3.0,
      "health": 20.0,
      "id": 24,
      "movement": 1,
      "name": "IceFortress",
      "range": 2,
      "skills": [
        "Escape",
        "Scout",
        "Skate",
        "Static"
      ]
    },
    {
      "attack": 1.0,
      "cost": 3,
      "defense": 1.0,
      "health": 10.0,
      "id": 25,
      "movement": 1,
      "name": "IceArcher",
      "range": 2,
      "skills": [
        "Dash",
        "Fortify",
        "Freeze",
        "Stiff"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 5.0,
      "health": 40.0,
      "id": 26,
      "movement": 2,
      "name": "Crab",
      "range": 1,
      "skills": [
        "Amphibious",
        "AutoFlood",
        "Escape",
        "Static"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 3.0,
      "health": 30.0,
      "id": 27,
      "movement": 1,
      "name": "Gaami",
      "range": 1,
      "skills": [
        "AutoFreeze",
        "FreezeArea",
        "Static"
      ]
    },
    {
      "attack": 3.0,
      "cost": 3,
      "defense": 1.0,
      "health": 5.0,
      "id": 28,
      "movement": 2,
      "name": "Hexapod",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Escape",
        "Sneak"
      ]
    },
    {
      "attack": 3.5,
      "cost": 10,
      "defense": 2.0,
      "health": 20.0,
      "id": 29,
      "movement": 3,
      "name": "Doomux",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Explode",
        "Static"
      ]
    },
    {
      "attack": 0.699999988079071,
      "cost": 3,
      "defense": 1.0,
      "health": 5.0,
      "id": 30,
      "movement": 2,
      "name": "Phychi",
      "range": 2,
      "skills": [
        "Dash",
        "DoubleAttack",
        "Fly",
        "Poison",
        "Surprise"
      ]
    },
    {
      "attack": 1.0,
      "cost": 3,
      "defense": 3.0,
      "health": 15.0,
      "id": 31,
      "movement": 1,
      "name": "Kiton",
      "range": 1,
      "skills": [
        "Creep",
        "Poison"
      ]
    },
    {
      "attack": 3.0,
      "cost": 8,
      "defense": 1.0,
      "health": 10.0,
      "id": 32,
      "movement": 1,
      "name": "Exida",
      "range": 3,
      "skills": [
        "Creep",
        "Poison",
        "Splash"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 3.0,
      "health": 20.0,
      "id": 33,
      "movement": 2,
      "name": "Centipede",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Eat",
        "Static"
      ]
    },
    {
      "attack": 2.0,
      "cost": 10,
      "defense": 2.0,
      "health": 10.0,
      "id": 34,
      "movement": 1,
      "name": "Segment",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Explode",
        "Independent",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 2.0,
      "health": 10.0,
      "id": 35,
      "movement": 3,
      "name": "Raychi",
      "range": 1,
      "skills": [
        "Dash",
        "Water"
      ]
    },
    {
      "attack": 1.0,
      "cost": 10,
      "defense": 1.0,
      "health": 10.0,
      "id": 36,
      "movement": 1,
      "name": "Shaman",
      "range": 1,
      "skills": [
        "Convert",
        "Static",
        "Swarm"
      ]
    },
    {
      "attack": 2.0,
      "cost": 2,
      "defense": 2.0,
      "health": 10.0,
      "id": 37,
      "movement": 1,
      "name": "Dagger",
      "range": 1,
      "skills": [
        "Dash",
        "Independent",
        "Static",
        "Surprise"
      ]
    },
    {
      "attack": 0.0,
      "cost": 8,
      "defense": 0.5,
      "health": 5.0,
      "id": 38,
      "movement": 2,
      "name": "Cloak",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Hide",
        "Infiltrate",
        "Scout",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 0.0,
      "cost": 2,
      "defense": 0.5,
      "health": 5.0,
      "id": 39,
      "movement": 2,
      "name": "Dinghy",
      "range": 1,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Hide",
        "Infiltrate",
        "Scout",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 2,
      "defense": 2.0,
      "health": 10.0,
      "id": 40,
      "movement": 2,
      "name": "Pirate",
      "range": 1,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Independent",
        "Static",
        "Surprise"
      ]
    },
    {
      "attack": 3.0,
      "cost": 15,
      "defense": 2.0,
      "health": 10.0,
      "id": 41,
      "movement": 2,
      "name": "Bomber",
      "range": 3,
      "skills": [
        "Carry",
        "Float",
        "Splash",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 5,
      "defense": 1.0,
      "health": 10.0,
      "id": 42,
      "movement": 3,
      "name": "Scoutship",
      "range": 2,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Scout",
        "Static"
      ]
    },
    {
      "attack": 0.0,
      "cost": 0,
      "defense": 1.0,
      "health": 10.0,
      "id": 43,
      "movement": 2,
      "name": "Raft",
      "range": 2,
      "skills": [
        "Carry",
        "Float",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 10.0,
      "id": 44,
      "movement": 3,
      "name": "Rammership",
      "range": 1,
      "skills": [
        "Carry",
        "Dash",
        "Float",
        "Static"
      ]
    },
    {
      "attack": 4.0,
      "cost": 10,
      "defense": 4.0,
      "health": 40.0,
      "id": 45,
      "movement": 2,
      "name": "Juggernaut",
      "range": 1,
      "skills": [
        "Carry",
        "Float",
        "Static",
        "Stiff",
        "Stomp"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 10.0,
      "id": 56,
      "movement": 2,
      "name": "Boomchi",
      "range": 1,
      "skills": [
        "Amphibious",
        "Dash",
        "Explode",
        "Stiff"
      ]
    },
    {
      "attack": 4.0,
      "cost": 20,
      "defense": 4.0,
      "health": 20.0,
      "id": 57,
      "movement": 2,
      "name": "LivingIsland",
      "range": 1,
      "skills": [
        "Algae",
        "Poison",
        "Static",
        "Stomp",
        "Water"
      ]
    },
    {
      "attack": 3.0,
      "cost": 5,
      "defense": 3.0,
      "health": 15.0,
      "id": 59,
      "movement": 1,
      "name": "Mantis",
      "range": 1,
      "skills": [
        "Creep",
        "Dash"
      ]
    },
    {
      "attack": 2.0,
      "cost": 0,
      "defense": 3.0,
      "health": 10.0,
      "id": 60,
      "movement": 0,
      "name": "InsectEgg",
      "range": 1,
      "skills": [
        "Explode",
        "Grow",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 5,
      "defense": 0.10000000149011612,
      "health": 10.0,
      "id": 61,
      "movement": 2,
      "name": "Moth",
      "range": 1,
      "skills": [
        "Dash",
        "Fly",
        "Infiltrate",
        "Poison",
        "Sneak",
        "Static",
        "Stiff"
      ]
    },
    {
      "attack": 2.0,
      "cost": 0,
      "defense": 2.0,
      "health": 10.0,
      "id": 62,
      "movement": 1,
      "name": "Larva",
      "range": 1,
      "skills": [
        "Creep",
        "Dash",
        "Grow",
        "Independent",
        "Static",
        "Surprise"
      ]
    }
  ]
}
"###;
use serde_json::{json,Value};
use std::io::{self,BufRead,Write};
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use crate::state::State;
use crate::rules::Rules;
use crate::geom::Geom;
use crate::core::Action;
fn int(v:&Value,k:&str)->i32{v[k].as_i64().unwrap_or(0) as i32}
fn flag(v:&Value,k:&str)->bool{v[k].as_bool().unwrap_or(false)}
fn arr<'a>(v:&'a Value,k:&str)->&'a Vec<Value>{v[k].as_array().unwrap()}
#[derive(Clone)]
struct Candidate {id:String,a:Action}
struct Bot {r:Rules,g:Geom,lastturn:i32,visits:Vec<f32>,prev:Vec<usize>,ecapital:Option<usize>,enemytech:[bool;128],actions_turn:i32,walls:Vec<bool>,last_hit:Option<(usize,f32,f32)>}
impl Bot {
 fn new(size:usize)->Self {Self{r:Rules::from_json(RULES_JSON).unwrap(),g:Geom::new(size),lastturn:-1,visits:vec![0.;size*size],prev:vec![0;size*size],ecapital:None,enemytech:[false;128],actions_turn:0,walls:vec![false;size*size],last_hit:None}}
 fn decode(&mut self,o:&Value)->(State,Vec<Candidate>) {
  let v=&o["visible_state"];
  if let Some((target,oldhp,predict))=self.last_hit.take(){if let Some(u)=arr(v,"units").iter().find(|u|int(u,"tile")==target as i32){let hp=u["health"].as_f64().unwrap_or(0.) as f32;if predict>1.&&oldhp-hp+1.5<predict && arr(v,"cities").iter().any(|c|int(c,"tile")==target as i32&&int(c,"owner")==int(u,"owner")){self.walls[target]=true;}}}
  let p=int(o,"player");let q=3-p;let mut s=State::empty(self.g.size);s.current_player=p;s.turn=int(v,"turn");s.reference_max_turns=200;
  s.players.tech[(p-1) as usize].fill(false);for x in arr(v,"technology"){s.players.tech[(p-1) as usize][x.as_u64().unwrap() as usize]=true;}
  s.players.tech[(q-1) as usize]=self.enemytech;s.players.tech[(q-1) as usize][0]=true;s.players.tech[(q-1) as usize][15]=true;
  s.players.stars[s.p()]=int(v,"stars");s.players.stars[(q-1) as usize]=20;
  for t in arr(v,"tiles"){let i=int(t,"tile") as usize;if flag(t,"known") {s.tiles.terrain[i]=int(t,"terrain");s.tiles.owner[i]=int(t,"owner");s.tiles.structure[i]=int(t,"structure");s.tiles.resource[i]=int(t,"resource");s.tiles.capital[i]=int(t,"capital");s.tiles.structure_level[i]=int(t,"structure_level");s.tiles.road[i]=int(t,"road")>0;s.tiles.explored[s.p()][i]=true;s.tiles.explored[(q-1) as usize][i]=true;if int(t,"capital")==q {self.ecapital=Some(i);}}else{s.tiles.terrain[i]=3;}}
  for c in arr(v,"cities"){let i=int(c,"tile") as usize;let owner=int(c,"owner");s.cities.owner[i]=owner;if owner==p {s.cities.level[i]=int(c,"level");s.cities.progress[i]=int(c,"progress");s.cities.population[i]=int(c,"population");s.cities.production[i]=int(c,"production");s.cities.connected[i]=flag(c,"connected");s.cities.riot[i]=flag(c,"riot");for (k,x) in arr(c,"rewards").iter().enumerate(){s.cities.rewards[i*9+k]=x.as_i64().unwrap() as i32;}}
   else if owner>0{s.cities.level[i]=s.tiles.structure_level[i].max(2);s.cities.production[i]=s.cities.level[i];if self.walls[i]{s.cities.rewards[i*9+1]=1;}}else{s.cities.level[i]=1;s.cities.production[i]=1;}
   s.cities.border[i]=if s.reward(i,5)>0{2}else{1};s.cities.order[i]=if s.tiles.capital[i]==owner{1}else{2+i as i32};
  }
  for t in 0..s.n {if s.tiles.owner[t]>0 {let owner=s.tiles.owner[t];let city=(0..s.n).filter(|&c|s.cities.owner[c]==owner).min_by_key(|&c|(self.g.d(c,t)>s.cities.border[c],self.g.d(c,t),s.cities.order[c]));if let Some(c)=city{s.tiles.city[t]=c as i32;s.cities.territory[c*s.n+t]=true;}}}
  for u in arr(v,"units"){let i=int(u,"tile") as usize;let k=int(u,"kind") as usize;let owner=int(u,"owner");s.units.owner[i]=owner;s.units.kind[i]=k as i32;s.units.health[i]=u["health"].as_f64().unwrap_or(10.) as f32;s.units.veteran[i]=flag(u,"veteran");s.units.moved[i]=flag(u,"moved");s.units.attacked[i]=flag(u,"attacked");s.units.kills[i]=int(u,"kills");s.units.home[i]=int(u,"home");s.units.passenger[i]=int(u,"passenger");s.units.poison[i]=flag(u,"poison");s.units.boosted[i]=flag(u,"boosted");s.units.invisible[i]=flag(u,"invisible");s.units.prev_x[i]=(i%s.size) as i32;s.units.prev_y[i]=(i/s.size) as i32;
   if owner==q {if self.r.skills[k][rules::CARRY] && k!=45 {if s.units.health[i]>20. {s.units.passenger[i]=12;}else if s.units.health[i]>15. {s.units.passenger[i]=11;}else if s.units.health[i]>10.+if s.units.veteran[i]{5.}else{0.} {s.units.passenger[i]=5;}else {s.units.passenger[i]=2;}}let tech=self.r.unit_tech[k];if tech<127{self.enemytech[tech as usize]=true;}if k==12{s.units.home[i]=0;}}
  }
  // Defenders' public tech is inferred conservatively from observed units, and normal midgame unlocks.
  if s.turn>12 {self.enemytech[20]=true;}if s.turn>18{self.enemytech[18]=true;}
  self.enemytech[0]=true;self.enemytech[15]=true;
  for (child,parent) in [(4,1),(3,2),(2,1),(5,4),(7,6),(8,6),(9,8),(13,10),(44,10),(14,44),(17,16),(18,15),(16,15),(23,20),(24,23),(22,21),(21,20),(38,7)] {if self.enemytech[child]{self.enemytech[parent]=true;}}
  s.players.tech[(q-1) as usize]=self.enemytech;
  for (key,dst) in [("peace",0),("known_players",1),("peace_offers",2)] {if let Some(a)=v["relations"][key].as_array(){for (k,x) in a.iter().enumerate(){let z=x.as_bool().unwrap_or(false);match dst{0=>{s.players.peace[s.p()][k]=z;s.players.peace[k][s.p()]=z;},1=>s.players.known[s.p()][k]=z,_=>s.players.peace_offer[s.p()][k]=z}}}}
  for (key,build) in [("monuments",false),("built",true)] {if let Some(a)=v["achievements"][key].as_array(){for x in a{let k=x.as_u64().unwrap() as usize;if build{s.players.built[s.p()][k]=true;}else{s.players.monuments[s.p()][k]=true;}}}}
  let legal=arr(o,"legal_actions").iter().map(|x|{let c=&x["command"];Candidate{id:x["action_id"].as_str().unwrap().to_owned(),a:Action{kind:int(c,"kind"),source:int(c,"source"),target:int(c,"target"),parameter:int(c,"parameter")}}}).collect();
  if s.turn!=self.lastturn {self.lastturn=s.turn;self.actions_turn=0;for x in &mut self.visits{*x*=0.83;}}
  (s,legal)
 }
}
fn unit_value(r:&Rules,s:&State,i:usize)->f32 {
 let k=s.units.kind[i] as usize;if k==0{return 0.;} let base=match k{2=>3.2,3=>4.4,4=>11.,5=>4.8,8=>11.,9=>4.5,10=>7.,11=>8.,12=>26.,38=>8.,41=>24.,42=>11.,43=>4.,44=>12.,45=>32.,_=>5.};
 let hp=s.units.health[i]/kernels::max_health(s,r,i);base*(0.18+0.82*hp.max(0.).powf(0.75))
}
struct Context {
 own:Vec<usize>,enemy:Vec<usize>,cities:Vec<usize>,neutral:Vec<usize>,ecities:Vec<usize>,
 enemy_reach:Vec<Vec<usize>>,potential:Vec<f32>,goal:Vec<usize>,goal_dist:Vec<f32>,goal_paths:Vec<Vec<f32>>,
 econ_threat:f32,water:f32,land_targets:usize,income:f32,
}
impl Bot {
 fn distances(&self,s:&State,k:usize,owner:i32,target:usize,naval:bool)->Vec<f32>{
  // Strategic Dijkstra, reversed edges, includes the expected cost of boarding at friendly ports.
  let mut d=vec![10000.;s.n];d[target]=0.;let mut heap=BinaryHeap::new();heap.push(Reverse((0i32,target)));
  while let Some(Reverse((cost,x)))=heap.pop(){if cost as f32>d[x]*10.+0.1{continue;}
   for &jj in &self.g.adj[x]{let j=jj as usize;let ter=s.tiles.terrain[x];let tfrom=s.tiles.terrain[j];let bridge=s.tiles.structure[x]==48;let fromwater=tfrom<=2&&s.tiles.explored[s.p()][j];let water=ter<=2&&s.tiles.explored[s.p()][x]&&!bridge;
    if ter==4&&!s.players.tech[(owner-1) as usize][20]{continue;}
    if water&&!naval&&!fromwater&&!(s.tiles.structure[x]==8&&s.tiles.owner[x]==owner){continue;}
    if ter==2&&!s.players.tech[(owner-1) as usize][13]&&!bridge{continue;}
    let roadx=s.tiles.road[x]||s.tiles.structure[x]==1||bridge;let roadj=s.tiles.road[j]||s.tiles.structure[j]==1||s.tiles.structure[j]==48;
    let speed=if water||fromwater{if k==43{2.}else{3.}}else{self.r.unit_movement[k].max(1) as f32};
    let step=if ter==4||ter==5{1.0}else if roadx&&roadj{0.5/speed}else{1.0/speed};
    let mut add=(step*100.).round() as i32;
    if s.units.owner[x]>0&&s.units.owner[x]!=owner{add+=if s.tiles.structure[x]==1{90}else{160};}
    if !s.tiles.explored[s.p()][x]{add+=25;}
    let new=cost+add;if (new as f32)<d[j]*10. {d[j]=new as f32/10.;heap.push(Reverse((new,j)));}
   }
  }d.iter_mut().for_each(|x|*x/=10.);d
 }
 fn context(&self,s:&State)->Context {
  let p=s.current_player;let mut c=Context{own:vec![],enemy:vec![],cities:vec![],neutral:vec![],ecities:vec![],enemy_reach:vec![],potential:vec![0.;s.n],goal:vec![0;s.n],goal_dist:vec![1000.;s.n],goal_paths:vec![vec![1000.;s.n];s.n],econ_threat:0.,water:0.,land_targets:0,income:0.};
  for i in 0..s.n {if s.units.owner[i]==p{c.own.push(i);}else if s.units.owner[i]>0{c.enemy.push(i);}
   if s.tiles.structure[i]==1{if s.tiles.owner[i]==p{c.cities.push(i);c.income+=s.cities.production[i] as f32+(s.tiles.capital[i]==p) as i32 as f32;}else if s.tiles.owner[i]==0{c.neutral.push(i);}else{c.ecities.push(i);}}
   if s.tiles.terrain[i]<=2&&s.tiles.explored[s.p()][i]{c.water+=1.;}
  }
  c.water/=s.tiles.explored[s.p()].iter().filter(|&&b|b).count().max(1) as f32;
  let mut ms=kernels::MoveScratch::new(s.n);
  for &e in &c.enemy {let mut v=vec![e];kernels::movement_flags(s,s.units.owner[e],&self.g,&mut ms.flags);kernels::reach(s,e,&self.r,&self.g,&mut ms,|t|v.push(t));c.enemy_reach.push(v);}
  // Create strategic objectives, including exploration frontiers rather than arbitrary unknown squares.
  let mut goals:Vec<(usize,f32)>=vec![];
  for &t in &c.neutral {if s.units.owner[t]!=p {goals.push((t,19.));}}
  for &t in &c.ecities {goals.push((t,22.));}
  for i in 0..s.n {if !s.tiles.explored[s.p()][i]{continue;}if s.tiles.structure[i]==2&&s.units.owner[i]!=p{goals.push((i,14.));}
   let range=if s.tiles.terrain[i]==4{2}else{1};let mut unknown:f32=0.;self.g.for_ball(i,range,|j|{if !s.tiles.explored[s.p()][j]{unknown+=1.;}});
   if unknown>0. && (s.tiles.terrain[i]>2||c.water>0.2) {let edge=(i%s.size).min(s.size-1-i%s.size).min((i/s.size).min(s.size-1-i/s.size));goals.push((i,4.0+unknown*0.65+edge.min(3) as f32*0.25));}
  }
  // Enemy units can be targets even when their cities are outside vision.
  for &e in &c.enemy {goals.push((e,if s.tiles.owner[e]==p{24.}else{12.}));}
  let mut cache=std::collections::HashMap::new();
  for &u in &c.own {
   let k=s.units.kind[u] as usize;let naval=self.r.skills[k][rules::FLOAT];let mut best:f32=-1.;
   for &(target,value) in &goals {
    let dt=cache.entry((k,target,naval)).or_insert_with(||self.distances(s,k,p,target,naval));
    let dd=dt[u];if dd>100.{continue;}
    let mut count:f32=0.;for &other in &c.own {if other==u{continue;}let od=dt[other];if od<dd-0.1 || (od<=dd+0.1&&other<u) {count+=if s.units.moved[other]{0.65}else{0.8};}}
    let saturation=if s.tiles.structure[target]==1&&s.tiles.owner[target]==0 {1.0+count*2.7}else if s.tiles.structure[target]==2{1.+count*3.0}else if s.tiles.structure[target]==1 {1.+(count-2.).max(0.)*0.14}else if s.units.owner[target]>0 {1.+(count-1.).max(0.)*0.22}else {1.+count*1.8};
    let score=value/(1.+dd*0.65)/saturation;
    if score>best {best=score;c.goal[u]=target;c.goal_dist[u]=dd;c.goal_paths[u]=dt.clone();}
   }
   if best>=0. {c.potential[u]=best;}
   let target=c.goal[u];if s.tiles.structure[target]==1&&s.tiles.owner[target]==0 && c.goal_dist[u]<20. {c.land_targets+=1;}
  }
  for &e in &c.enemy {let dist=c.cities.iter().map(|&t|self.g.d(t,e)).min().unwrap_or(99);if dist<=4{c.econ_threat+=(5-dist) as f32*unit_value(&self.r,s,e)/15.;}}
  c
 }
 fn incoming(&self,s:&State,c:&Context,t:usize,hp:f32,kind:usize,ignore:Option<usize>)->f32 {
  if hp<=0. {return 0.;}let p=s.current_player;
  let mut temp=s.clone();temp.units.owner[t]=p;temp.units.kind[t]=kind as i32;temp.units.health[t]=hp;
  let mut attacks=vec![];
  for (idx,&e) in c.enemy.iter().enumerate() {
   if ignore==Some(e)||s.units.owner[e]==0||s.units.owner[e]==p{continue;}
   let ek=s.units.kind[e] as usize;
   let range=self.r.unit_range[ek];let dash=self.r.dash[ek];
   let can=if dash{c.enemy_reach[idx].iter().any(|&x|x!=t&&self.g.d(x,t)<=range)}else{self.g.d(e,t)<=range};
   if !can||(self.r.unit_attack[ek]<=0.&&ek!=10) {continue;}if ek==10{return unit_value(&self.r,&temp,t)*1.8;}
   let d=kernels::combat(self.r.unit_attack[ek],s.units.health[e],kernels::max_health(s,&self.r,e),self.r.unit_defense[kind],hp,kernels::max_health(&temp,&self.r,t),military::defense_bonus(&temp,&self.r,t)).0;
   attacks.push((d,e));
  }
  attacks.sort_by(|a,b|b.0.partial_cmp(&a.0).unwrap());
  let mut health=hp;let mut loss:f32=0.;
  for (_,e) in attacks {let ek=s.units.kind[e] as usize;
   if ek==10{return unit_value(&self.r,&temp,t)*1.6;}
   let (d,ret,_)=kernels::combat(self.r.unit_attack[ek],s.units.health[e],kernels::max_health(s,&self.r,e),self.r.unit_defense[kind],health,kernels::max_health(&temp,&self.r,t),military::defense_bonus(&temp,&self.r,t));
   // If attacking this unit means dying to others or leaving a city, enemy may not choose it.
   let defenders=c.own.iter().filter(|&&f|f!=t&&self.g.d(f,e)<=2).count();
   let probability=if defenders>=2&&ret>=s.units.health[e]*0.6&&self.r.unit_range[ek]<=1{0.65}else{1.0};
   health-=d*probability;loss+=d*probability;
   if health<=0.{return unit_value(&self.r,&temp,t)+2.5;}
  }
  unit_value(&self.r,&temp,t)*(loss/hp).min(1.)*0.78
 }
 fn knight_chain(&self,s:&State,at:usize,depth:i32)->f32 {
  if depth<=0{return 0.;}let mut targets=vec![];military::attack_targets(s,at,&self.r,&self.g,|j|targets.push(j));let mut best:f32=0.;
  targets.sort_by(|&a,&b|unit_value(&self.r,s,b).partial_cmp(&unit_value(&self.r,s,a)).unwrap());
  for t in targets.into_iter().take(4) {let dk=s.units.kind[t] as usize;let damage=kernels::combat(3.5,s.units.health[at],kernels::max_health(s,&self.r,at),self.r.unit_defense[dk],s.units.health[t],kernels::max_health(s,&self.r,t),military::defense_bonus(s,&self.r,t)).0;
   if damage>=s.units.health[t]{let mut n=s.clone();military::attack(&mut n,at,t,&self.r,&self.g);let x=if n.units.owner[t]==s.current_player{t}else{at};best=best.max(unit_value(&self.r,s,t)*4.+5.+self.knight_chain(&n,x,depth-1));}
  }best
 }
 fn attack_score(&self,s:&State,c:&Context,src:usize,dst:usize)->f32 {
  let k=s.units.kind[src] as usize;let dk=s.units.kind[dst] as usize;
  if k==10{return 10.+unit_value(&self.r,s,dst)*2.;}
  if k==38||k==39{return 12.;}
  let mh=kernels::max_health(s,&self.r,src);let dh=kernels::max_health(s,&self.r,dst);
  let (damage,ret,_) = kernels::combat(self.r.unit_attack[k],s.units.health[src],mh,self.r.unit_defense[dk],s.units.health[dst],dh,military::defense_bonus(s,&self.r,dst));
  let kill=damage>=s.units.health[dst];let retaliation=if self.r.stiff[dk]||self.g.d(src,dst)>self.r.unit_range[dk]||self.r.skills[k][rules::SURPRISE]{0.}else{ret};
  let mut after=s.clone();military::attack(&mut after,src,dst,&self.r,&self.g);
  let at=if after.units.owner[src]==s.current_player{src}else if after.units.owner[dst]==s.current_player{dst}else{src};
  let gain=unit_value(&self.r,s,dst)-if kill{0.}else{unit_value(&self.r,&after,dst)};
  let ownloss=unit_value(&self.r,s,src)-if after.units.owner[at]==s.current_player{unit_value(&self.r,&after,at)}else{0.};
  let mut score=gain*5.-ownloss*4.+if kill{4.0}else{0.};
  if !kill&&after.units.owner[dst]>0 {
   let mut finish=0.;
   for &u in &c.own {if u==src||s.units.attacked[u]||s.units.owner[u]!=s.current_player {continue;}let uk=s.units.kind[u] as usize;let distance=self.g.d(u,dst);if distance<=self.r.unit_range[uk]&&(!s.units.moved[u]||self.r.dash[uk]) {
    let dm=kernels::combat(self.r.unit_attack[uk],s.units.health[u],kernels::max_health(s,&self.r,u),self.r.unit_defense[dk],after.units.health[dst],dh,military::defense_bonus(&after,&self.r,dst)).0;
    finish+=dm;
   }}
   if finish>=after.units.health[dst] {score+=unit_value(&self.r,s,dst)*1.5+4.;}
  }
  if kill&&k==4{score+=self.knight_chain(&after,at,3);}
  if after.units.owner[at]==s.current_player {
   let risk=self.incoming(&after,c,at,after.units.health[at],k,if kill{Some(dst)}else{None});
   let oldrisk=self.incoming(s,c,src,s.units.health[src],k,None);
   score-=risk*2.1-oldrisk.min(risk)*1.0;
  }
  if s.tiles.structure[dst]==1 {
   if s.tiles.owner[dst]==s.current_player{score+=if kill{38.}else{12.};}
   else if s.tiles.owner[dst]>0{score+=if kill&&self.r.unit_range[k]==1{30.}else{9.};}
  }
  if s.tiles.structure[src]==1&&s.tiles.owner[src]!=s.current_player&&s.tiles.owner[src]>0&&!kill {score-=10.;}
  if self.r.skills[k][rules::SPLASH]{for &j in &self.g.adj[dst]{let j=j as usize;if s.units.owner[j]>0&&s.units.owner[j]!=s.current_player{score+=unit_value(&self.r,s,j)*1.1;}}}
  // Full healthy units should attack first; retain injured finishing blows when available.
  score+=damage.min(s.units.health[dst])*0.08-retaliation*0.04;
  score
 }
}
impl Bot {
 fn pop_score(&self,s:&State,c:&Context,t:usize,pop:f32,cost:f32)->f32 {
  let city=s.tiles.city[t];if city<0{return -100.;}let city=city as usize;if s.cities.owner[city]!=s.current_player{return -100.;}
  let level=s.cities.level[city];let need=(level+1-s.cities.progress[city]) as f32;
  let mut available:f32=0.;let mut prices=vec![];
  for j in 0..s.n {if s.tiles.city[j]!=city as i32||s.tiles.owner[j]!=s.current_player||s.tiles.structure[j]!=0{continue;}
   let res=s.tiles.resource[j];let tech=&s.players.tech[s.p()];if (res==1&&tech[15])||(res==3&&tech[10])||(res==6&&tech[6]){prices.push(2.);available+=1.;}
   if s.tiles.terrain[j]==5&&tech[16]{prices.push(3.);available+=1.;}
   if res==2&&tech[8]{prices.extend([2.5,2.5]);available+=2.;}
   if res==5&&tech[23]{prices.extend([2.5,2.5]);available+=2.;}
  }
  let mut score:f32=13.+pop*7.-cost*1.5;
  let cheap=(cost/pop.max(1.)).max(1.);score+=5./cheap;
  if need<=pop {score+=match level{1=>32.,2=>25.,3=>19.,_=>45.};}
  else if available>=need {score+=match level{1=>24.,2=>17.,3=>11.,_=>22.}/(need-pop+1.).sqrt();}
  else{score-=7.;}
  if level>=4{score+=12.;}if level==3&&available>=need{score+=7.;}if level>=7{score-=9.;}
  if c.econ_threat>1.&&need>pop {score-=c.econ_threat.min(6.)*2.;}
  if s.tiles.structure[t]==0&&s.tiles.resource[t]>0&&cost>2.&&s.tiles.resource[t]!=2&&s.tiles.resource[t]!=5 {score-=7.;}
  if s.units.owner[city]>0&&s.units.owner[city]!=s.current_player {if level>=4&&need<=pop{score+=180.;}else{score-=35.;}}
  if s.turn>160{score-=8.;}score
 }
 fn research_score(&self,s:&State,c:&Context,tech:usize)->f32 {
  let cities=c.cities.len() as i32;let cost=kernels::research_cost(s,&self.r,tech) as f32;let stars=s.players.stars[s.p()] as f32;
  let mut animals:f32=0.;let mut fruits:f32=0.;let mut fish:f32=0.;let mut forests:f32=0.;let mut farms:f32=0.;let mut mines:f32=0.;let mut mountains:f32=0.;
  for i in 0..s.n {if s.tiles.terrain[i]==4&&s.tiles.explored[s.p()][i]{mountains+=1.;}
   if s.tiles.owner[i]!=s.current_player||s.tiles.structure[i]!=0{continue;}
   match s.tiles.resource[i]{1=>animals+=1.,6=>fruits+=1.,3=>fish+=1.,2=>farms+=1.,5=>mines+=1.,_=>()}
   if s.tiles.terrain[i]==5{forests+=1.;}
  }
  let ranged=c.enemy.iter().filter(|&&e|matches!(s.units.kind[e],8|9|4)).count();let foes=c.enemy.len() as f32;let units=c.own.len() as f32;let mut score:f32=-50.;let surplus=stars-cost;
  let navy=c.own.iter().filter(|&&u|self.r.skills[s.units.kind[u] as usize][rules::FLOAT]).count();
  let ports=(0..s.n).filter(|&t|s.tiles.structure[t]==8&&s.tiles.owner[t]==s.current_player).count();
  let noexpansion=c.land_targets==0&&c.neutral.len()<2;
  let military= c.econ_threat>0.4 || s.turn>10;
  match tech {
   1=> {score=if s.turn>=1 {28.}else{-10.};if units<3.&&cities<=2 {score+=7.;}if c.water>0.45{score-=6.;}if surplus<3.&&s.turn<3 {score-=12.;}}
   6=>{score=if fruits>=1.{12.+fruits.min(5.)*3.8}else{-15.};if surplus<2.{score-=15.;}if cities==1&&fruits>=2.&&s.turn>=2{score+=8.;}}
   10=>{score=if fish>0.{9.+fish.min(5.)*3.4}else{-12.};if c.water>0.25&&s.turn>=4{score+=15.;}if noexpansion&&c.water>0.18{score+=14.;}if surplus<2.{score-=14.;}}
   16=>{score=if forests>=2.{19.+forests.min(12.)*2.0}else{-15.};if surplus<3. {score-=13.;}if cities<=2&&s.turn>=4&&forests>=4.{score+=7.;}}
   18=>{score=if military{29.}else{-10.};if foes>=2. {score+=22.;}if c.enemy.iter().any(|&e|s.units.kind[e]==9||s.units.kind[e]==8){score+=10.;}if foes>=2. {score+=7.;}if units>=3.{score+=4.;}if s.players.tech[s.p()][24]||s.players.tech[s.p()][17]{score-=10.;}if surplus<3.{score-=18.;}}
   20=>{score=8.+mountains.min(18.)*0.65;if s.turn<3{score-=12.;}if noexpansion {score+=6.;}if s.turn>8{score+=7.;}if surplus<3.{score-=10.;}}
   17=>{score=if c.income>=7.&&s.turn>=5{31.}else{-15.};let huts=(0..s.n).filter(|&i|s.tiles.owner[i]==s.current_player&&s.tiles.structure[i]==12).count();if huts>=4{score+=13.;}if c.enemy.iter().any(|&u|s.units.kind[u]==12||s.units.kind[u]==5){score+=8.;}if s.turn>20&&c.income>=8.{score+=8.;}if surplus<5.{score-=16.;}}
   23=>{score=10.+mines.min(4.)*6.;if c.enemy.iter().any(|&e|s.units.kind[e]==11){score+=15.;}if c.income>=9.&&s.turn>10 {score+=12.;}if surplus<5.{score-=15.;}}
   24=>{score=if c.income>=8.&&military{43.}else{2.};if surplus<5.{score-=20.;}if foes>2. {score+=6.;}}
   2=>{score=if c.income>=13.&&s.turn>12{25.}else{-15.};if ranged>=3&&c.income>=9.{score=49.;}if units>foes*2.5&&foes>=2.{score+=4.;}if surplus<5.{score-=18.;}}
   3=>{score=if c.income>=12.&&s.turn>12{39.}else{0.};if ranged>=3&&c.income>=9.{score=55.;}if c.enemy.iter().filter(|&&e|matches!(s.units.kind[e],3|8|9|4)).count()>=3 {score+=14.;}if surplus<8.{score-=12.;}}
   4=>{score=if cities>=2 {17.}else{1.};if c.income>10.&&s.turn>7 {score+=6.;}if surplus<6.{score-=12.;}}
   8=>{score=farms.min(6.)*4.+4.;if surplus<5.{score-=20.;}}
   9=>{score=if farms>=2.{17.}else{-5.};if surplus<10.{score-=20.;}}
   13=>{score=if ports>0||navy>0{29.}else{if c.water>0.25&&s.turn>9&&c.ecities.is_empty(){21.}else{-10.}};if noexpansion&&c.water>0.2 {score+=6.;}if surplus<5.{score-=14.;}}
   44=>{score=if navy>0{30.}else if ports>0{18.}else{-12.};if c.water>0.3&&s.turn>15 {score+=5.;}if surplus<5.{score-=15.;}}
   14=>{score=if navy>=1&&c.income>=9.{42.}else{-25.};if c.enemy.iter().any(|&e|s.units.kind[e]>=41){score+=13.;}if c.water>0.35&&s.turn>20{score+=7.;}if surplus<15.{score-=18.;}}
   7=>{score=if c.econ_threat>2.&& !s.players.tech[s.p()][24]{21.}else{-12.};if c.water>0.35&&ports>0 {score+=10.;}if surplus<3.{score-=20.;}}
   12=>{score=if navy>=3{24.}else{-20.};if surplus<15.{score-=20.;}}
   5=>{let saw=(0..s.n).filter(|&i|s.tiles.owner[i]==s.current_player&&matches!(s.tiles.structure[i],13|6|22)).count();score=if saw>=2&&s.turn>15{24.}else{-20.};if surplus<10.{score-=20.;}}
   _=>{}
  }
  if c.econ_threat>3.&&surplus<6.&& !matches!(tech,18|24|3|17|7){score-=18.;}
  if stars>cost+c.income*3.+20. {score+=12.;}
  score-cost*0.22
 }
 fn train_score(&self,s:&State,c:&Context,t:usize,k:usize)->f32 {
  let stars=s.players.stars[s.p()] as f32;let cost=self.r.unit_cost[k] as f32;
  let nearest=c.enemy.iter().map(|&e|self.g.d(e,t)).min().unwrap_or(99);
  let frontline=c.ecities.iter().map(|&e|self.g.d(e,t)).min().unwrap_or(nearest).min(nearest);
  let local=c.own.iter().filter(|&&u|self.g.d(u,t)<=3).count() as f32;
  let foes=c.enemy.iter().filter(|&&e|self.g.d(e,t)<=4).count() as f32;
  let cnt=c.own.len() as f32;let cities=c.cities.len() as f32;
  if s.units.kind[t]==43 {
   let mut score=match k{41=>if c.income>=10.{35.}else{19.},42=>26.,44=>27.,_=>-99.};
   if k==41&&nearest<=2{score-=12.;}if k==44&&s.units.health[t]>=15.{score+=6.;}if k==42&&s.units.health[t]>15.{score-=2.;}
   if c.own.iter().filter(|&&u|s.units.kind[u]==41).count()<2&&k==41{score+=6.;}
   if k==42&&s.players.tech[s.p()][44]{score-=4.;}
   if k==41&&s.units.health[t]<=7.{score-=15.;}return score;
  }
  let needexplore=(c.neutral.len()+2) as f32;
  let mut score=match k {
   2=> if cnt<needexplore.min(cities*1.8+1.) {25.}else{5.},
   3=> if cnt<needexplore.min(cities*1.6+2.){30.}else{9.},
   9=> if nearest<7{29.}else{8.},
   11=>if frontline<10{33.}else{17.},
   8=>if frontline<9 {29.}else{12.},
   4=>if frontline<10 {30.}else{16.},
   5=>if nearest<=3 {23.}else{-10.},
   10=>-25.,38=>-35.,_=>-50.
  };
  if c.econ_threat>0. && nearest<=4 {score+=13.+foes*2.-local*1.5;}
  if nearest<=2 {score+=12.;if k==5{score+=8.;}if k==8 {score-=14.;}if k==2{score-=6.;}}
  if cnt>cities*3.+2. {score-=12.;}
  if local>4.{score-=(local-4.)*3.;}
  if k==2&&s.players.tech[s.p()][1]&&nearest>2{score-=7.;}
  if k==2&&s.players.tech[s.p()][24]{score-=9.;}
  if k==3&&s.players.tech[s.p()][24]&&nearest<6{score-=12.;}if k==3&&s.players.tech[s.p()][18]&&nearest<6{score-=10.;}
  if k==9&&s.players.tech[s.p()][24]{score-=3.;}
  if k==8 {let existing=c.own.iter().filter(|&&u|s.units.kind[u]==8&&self.g.d(u,t)<5).count();score-=existing as f32*10.;if cnt<4.{score-=12.;}}
  if k==4 {let existing=c.own.iter().filter(|&&u|s.units.kind[u]==4).count();score-=existing as f32*1.5;if c.enemy.iter().filter(|&&e|matches!(s.units.kind[e],3|8|9)).count()>=3{score+=8.;}}
  if nearest==99&&c.neutral.is_empty()&&c.water<0.15&&s.turn>10{score-=10.;}
  if c.water>0.3 {if k==5&&c.own.iter().any(|&u|s.tiles.structure[u]==8||s.units.kind[u]>=41){score+=10.;}if k==8 {score-=7.;}}
  if stars-cost<3.&&c.econ_threat<1.{score-=5.;}
  if stars>c.income*3.+15. {score+=6.;}
  let risk=self.incoming(s,c,t,self.r.unit_health[k],k,None);
  score-=risk*1.0;
  if s.turn==0{score-=15.;}
  score
 }
 fn move_score(&self,s:&State,c:&Context,src:usize,dst:usize)->f32 {
  let p=s.current_player;let k=s.units.kind[src] as usize;let hp=s.units.health[src];
  if s.tiles.structure[src]==1&&s.tiles.owner[src]!=p&&s.units.moved[src]==false&&s.units.attacked[src]==false{return -100.;}
  let target=c.goal[src];let naval=self.r.skills[k][rules::FLOAT];let dt=&c.goal_paths[src];
  let progress=dt[src]-dt[dst];
  let mut score=if dt[src]<100.{(progress*10.).min(22.)+c.potential[src]*0.45}else{-5.};
  if progress<0. {score-=6.;}if progress.abs()<0.01{score-=5.;}
  let oldrisk=self.incoming(s,c,src,hp,k,None);
  let mut after=s.clone();military::relocate(&mut after,src,dst,&self.r,&self.g,false,None);
  let nk=after.units.kind[dst] as usize;let risk=self.incoming(&after,c,dst,hp,nk,None);
  score-=risk*3.4;score+=oldrisk.min(20.)*1.7;
  let mut attack:f32=-100.;if !after.units.attacked[dst] {military::attack_targets(&after,dst,&self.r,&self.g,|e|{let x=self.attack_score(&after,c,dst,e);attack=attack.max(x);});}
  if attack>0. {score+=attack*0.75+5.;}
  let city=s.tiles.structure[dst]==1&&s.tiles.owner[dst]!=p;
  if city {score+=if s.tiles.owner[dst]==0 {43.}else{60.};if risk>=unit_value(&self.r,&after,dst){score-=25.;}}
  if s.tiles.structure[dst]==2{score+=30.;}
  if k==43&&s.tiles.terrain[dst]>2&&s.tiles.structure[dst]!=8 {score+=2.;}
  if nk==43&&k!=43&&c.water>0.15 {score+=2.;}
  if s.tiles.structure[dst]==8&&nk==45 {score+=8.;}
  let range=if s.tiles.terrain[dst]==4||self.r.skills[k][rules::SCOUT]{2}else{1};let mut explore:f32=0.;self.g.for_ball(dst,range,|j|{if !s.tiles.explored[s.p()][j]{explore+=1.;}});
  score+=explore*1.0;
  score-=self.visits[dst]*1.6;
  let close=c.own.iter().filter(|&&u|u!=src&&self.g.d(u,dst)<=1).count() as f32;
  if !city{score-=close*0.4;}
  if s.tiles.structure[src]==1&&s.tiles.owner[src]==p {score+=if c.own.len()<(c.cities.len()*3+2){4.0}else{0.};if oldrisk>0.&&c.enemy.iter().any(|&e|self.g.d(src,e)<=2) {score-=if c.cities.len()==1{40.}else{25.};}}
  if s.tiles.structure[dst]==1&&s.tiles.owner[dst]==p&&!city {score-=6.;}
  if hp<kernels::max_health(s,&self.r,src)*0.65&&risk>0.&&attack<15. {score-=10.;}
  // Ranged siege engines want to settle into firing distance instead of touching defenders.
  if matches!(k,8|41|9|42) {let near=c.enemy.iter().map(|&e|self.g.d(e,dst)).min().unwrap_or(99);if near<=1 {score-=if k==9||k==42{8.}else{18.};}if near<=3 {score+=8.;}if near==2 {score+=3.;}}
  // Free the deployment city, but don't wander from defensible positions for no purpose.
  score
 }
}
impl Bot {
 fn building_score(&self,s:&State,c:&Context,t:usize,k:usize)->f32 {
  let p=s.current_player;
  match k {
   12=>{let adj=self.g.adj[t].iter().filter(|&&j|s.tiles.owner[j as usize]==p&&s.tiles.structure[j as usize]==13).count();self.pop_score(s,c,t,1.+adj as f32,3.)}
   5|21=>self.pop_score(s,c,t,2.,5.),
   6|13|22=> {let req=if k==6{5}else if k==13{12}else{21};let count=self.g.adj[t].iter().filter(|&&j|s.tiles.owner[j as usize]==p&&s.tiles.structure[j as usize]==req).count();let pop=count as f32*if k==22{2.}else{1.};let mut score=self.pop_score(s,c,t,pop,5.);if count<2{score-=13.;}score+pop*1.5}
   23..=29=>self.pop_score(s,c,t,3.,0.)+25.,
   8=>{
    let ports:Vec<usize>=(0..s.n).filter(|&i|s.tiles.structure[i]==8&&s.tiles.owner[i]==p).collect();
    if ports.iter().any(|&q|self.g.d(q,t)<=3){return -25.;}
    let adjacent=c.own.iter().filter(|&&u|self.g.d(u,t)<=2&&s.units.kind[u]<40).count();
    let mut water_size:f32=0.;let mut ocean:f32=0.;for i in 0..s.n{if self.g.d(i,t)<=3&&s.tiles.explored[s.p()][i] {if s.tiles.terrain[i]<=2 {water_size+=1.;if s.tiles.terrain[i]==2{ocean+=1.;}}}}
    let mut score=if ports.is_empty()&&c.water>0.2&&s.turn>5{25.}else{5.};
    score+=water_size.min(22.)*0.28+ocean.min(10.)*0.25;
    if adjacent==0{score-=10.;}else{score+=3.;}
    if c.land_targets==0 {score+=12.;}
    if s.turn<6 {score-=15.;}
    if s.players.stars[s.p()]<12 {score-=8.;}
    let enemy_target=self.ecapital.or_else(||c.ecities.first().copied());if let Some(e)=enemy_target {let dist=self.g.d(e,t);score+=((s.size as i32-dist).max(0) as f32)*0.6;}
    if s.tiles.resource[t]==3{score-=8.;}
    if c.econ_threat>1.&& !c.ecities.is_empty(){score-=22.;}score
   }
   3=>{
    // Roads connect cities through a minimal path; only build if on that path.
    let cap=c.cities.iter().find(|&&c|s.tiles.capital[c]==p).copied().unwrap_or(c.cities[0]);
    let mut road_dist=vec![9999i32;s.n];let mut parent=vec![usize::MAX;s.n];let mut heap=BinaryHeap::new();road_dist[cap]=0;heap.push(Reverse((0,cap)));
    while let Some(Reverse((d,x)))=heap.pop(){if d>road_dist[x]{continue;}
     for &j in &self.g.adj[x]{let j=j as usize;if !s.tiles.explored[s.p()][j]||s.tiles.terrain[j]<=2||s.tiles.terrain[j]==4||(s.tiles.owner[j]>0&&s.tiles.owner[j]!=p){continue;}
      let add=if s.tiles.road[j]||s.tiles.structure[j]==1{0}else{1};let nd=d+add;if nd<road_dist[j]{road_dist[j]=nd;parent[j]=x;heap.push(Reverse((nd,j)));}
     }
    }
    let mut score:f32=-35.;for &ct in &c.cities {if ct==cap||s.cities.connected[ct]||road_dist[ct]>5{continue;}let mut at=ct;for _ in 0..s.n{if at==cap||at==usize::MAX{break;}if at==t{score=score.max(24.-road_dist[ct] as f32*2.);break;}at=parent[at];}}
    if s.players.stars[s.p()]<6{score-=12.;}score
   }
   50=>{let mut count:f32=0.;for &j in &self.g.adj[t]{let j=j as usize;if s.tiles.owner[j]==p&&matches!(s.tiles.structure[j],6|13|22){count+=s.tiles.structure_level[j] as f32;}}if count>=2.{24.+count*2.}else{-10.}}
   48=>{
    let mut count=0;for &j in &self.g.adj[t]{if s.tiles.terrain[j as usize]>2&&s.tiles.explored[s.p()][j as usize]{count+=1;}}if count>=4&&c.own.iter().any(|&u|self.g.d(u,t)==1)&&c.water>0.15&&c.land_targets==0{12.}else{-30.}
   }
   _=>-100.
  }
 }
 fn choose(&mut self,o:&Value)->String{
  let (s,legal)=self.decode(o);let c=self.context(&s);let mut best=-1e10;let mut selected=0;
  for (idx,a) in legal.iter().enumerate(){let ac=a.a;let k=ac.kind;let src=ac.source.max(0) as usize;let dst=ac.target.max(0) as usize;let param=ac.parameter.max(0) as usize;
   let mut score=match k {
    11=>-1e8,10=>0.,
    9=>match param{3=>1000.,4=>970.,7=>1000.,1=>960.,6=>1100.,2=>900.,8=>1000.,5=>{
     let mut resources:f32=0.;let mut free:f32=0.;let t=dst;for i in 0..s.n{if self.g.d(t,i)==2&&s.tiles.owner[i]==0{free+=1.;if s.tiles.resource[i]>0||s.tiles.terrain[i]==5{resources+=1.;}}}
     if resources>=6.||free>=15.{1010.+resources}else{980.}
    },_=>900.},
    8=>if s.tiles.structure[src]==1{if s.tiles.owner[src]>0{1500.}else{300.}}else{210.},
    2=>self.attack_score(&s,&c,src,dst)+36.,
    1=>self.move_score(&s,&c,src,dst),
    4=>self.train_score(&s,&c,src,param),
    5=>self.pop_score(&s,&c,dst,1.,2.),
    6=>self.building_score(&s,&c,dst,param),
    7=>self.research_score(&s,&c,param),
    3=>match param{
     16=>180.,
     7=>{let missing=kernels::max_health(&s,&self.r,src)-s.units.health[src];let risk=self.incoming(&s,&c,src,s.units.health[src],s.units.kind[src] as usize,None);
      let heal=if s.tiles.owner[src]==s.current_player{4.}else{2.};
      let max=kernels::max_health(&s,&self.r,src);
      let mut score=if missing>=heal {10.+heal*1.5}else{1.+missing};
      if s.units.health[src]<max*0.55{score+=12.;}
      if s.units.kind[src]==12 {score+=8.;}
      if risk>unit_value(&self.r,&s,src)*0.7 {score-=8.;}
      score
     },
     9=>{let mut gain:f32=0.;for &j in &self.g.adj[src]{let j=j as usize;if s.units.owner[j]==s.current_player{gain+=(kernels::max_health(&s,&self.r,j)-s.units.health[j]).min(4.);}}gain*4.+10.},
     2=>{if s.tiles.resource[dst]==1||self.g.adj[dst].iter().any(|&j|s.tiles.structure[j as usize]==13){-60.}else{let city=s.tiles.city[dst];if city>=0&&s.cities.level[city as usize]>=4{-25.}else if s.players.stars[s.p()]<3{10.}else{-12.}}},
     19=>if ac.target==0{2.}else{-50.},
     18=>if c.own.len()>c.enemy.len()*2+3{10.}else{-20.},
     _=>-100.
    },
    _=>-100.
   };
   // Use the catalog's guaranteed actions; never risk simulation-legality assumptions.
   if k==2 && score<34. {score-=36.;}
   if self.actions_turn>160&&k!=9&&k!=8&&k!=10&&k!=2 {score=-10.;}
   score+=((idx*73%37) as f32)*0.0001;
   if score>best {best=score;selected=idx;}
  }
  let ac=legal[selected].a;
  if ac.kind==2{let src=ac.source as usize;let dst=ac.target as usize;let k=s.units.kind[src] as usize;let dk=s.units.kind[dst] as usize;
   if s.tiles.structure[dst]==1&&s.tiles.owner[dst]!=s.current_player {let predicted=kernels::combat(self.r.unit_attack[k],s.units.health[src],kernels::max_health(&s,&self.r,src),self.r.unit_defense[dk],s.units.health[dst],kernels::max_health(&s,&self.r,dst),military::defense_bonus(&s,&self.r,dst)).0;self.last_hit=Some((dst,s.units.health[dst],predicted));}
  }
  if ac.kind==1{self.visits[ac.source as usize]+=0.4;self.visits[ac.target as usize]+=0.5;}self.actions_turn+=1;
  legal[selected].id.clone()
 }
}
fn main(){let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout());let mut bot:Option<Bot>=None;
 for line in stdin.lock().lines(){let l=match line{Ok(l)=>l,Err(_)=>break};if l.is_empty(){continue;}let obs:Value=serde_json::from_str(&l).unwrap();let size=int(&obs["visible_state"],"board_size") as usize;if bot.is_none(){bot=Some(Bot::new(size));}let id=bot.as_mut().unwrap().choose(&obs);writeln!(stdout,"{}",json!({"action_id":id})).unwrap();stdout.flush().unwrap();}
}
