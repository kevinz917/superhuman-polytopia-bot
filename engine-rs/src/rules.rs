//! Rule tables, built from gpu/rules.json exactly as gpu/rules.py::load_rules("native") does.
//! Nothing here is re-typed by hand; the JSON is the single source.

use serde::Deserialize;

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

#[derive(Deserialize)]
struct UnitRow {
    id: usize,
    name: String,
    health: f32,
    attack: f32,
    defense: f32,
    movement: i32,
    range: i32,
    cost: i32,
    skills: Vec<String>,
}

#[derive(Deserialize)]
struct TechRow {
    id: i64,
    tier: i32,
    requires: Option<i64>,
    next: Vec<i64>,
    structures: Vec<usize>,
}

#[derive(Deserialize)]
struct StructRow {
    id: usize,
    cost: Option<i32>,
    population: Option<i32>,
    score: Option<i32>,
    stars: Option<i32>,
    resource: Option<i32>,
    terrain: Vec<usize>,
    adjacent: Vec<usize>,
    per_city: bool,
    per_tribe: bool,
    tribe: Option<i64>,
}

#[derive(Deserialize)]
struct Table {
    engine_revision: String,
    units: Vec<UnitRow>,
    technologies: Vec<TechRow>,
    structures: Vec<StructRow>,
}

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
    /// The tribe's tech list: the techs reachable from Basic in breadth-first order over `next` (the
    /// game's tech unlocks, in its order), each with its `next` list. The unlockable-tech walk uses it.
    pub tech_tree: Vec<(usize, Vec<usize>)>,
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
        let data: Table = serde_json::from_str(text).map_err(|e| format!("rules.json: {e}"))?;
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
            tech_tree: Vec::new(),
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
        let mut next: Vec<Vec<usize>> = vec![Vec::new(); 128];
        for t in &data.technologies {
            if (0..128).contains(&t.id) {
                next[t.id as usize] = t.next.iter().filter(|&&k| (0..128).contains(&k)).map(|&k| k as usize).collect();
            }
        }
        // at most 200 dequeues, as the game
        let (mut queue, mut at) = (vec![0usize], 0usize);
        while at < queue.len() && at < 200 {
            let k = queue[at];
            at += 1;
            if r.tech_exists[k] && !r.tech_tree.iter().any(|(t, _)| *t == k) {
                r.tech_tree.push((k, next[k].clone()));
                queue.extend(next[k].iter().copied());
            }
        }
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
