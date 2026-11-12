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
    /// Elyrion Grow: the unit whose upgrades from is this one, and growth rate (turns)
    grows_into: Option<usize>,
    growth_rate: Option<i32>,
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
    /// Elyrion: enchant animal's `creates` unit; Sanctuary's adjacency requirements terrain, its adjacency improvements
    /// resource (the level), max level and growth rate (Attract)
    creates_unit: Option<usize>,
    #[serde(default)]
    adjacent_terrain: Vec<usize>,
    level_resource: Option<i32>,
    max_level: Option<i32>,
    growth_rate: Option<i32>,
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
    /// Elyrion: creates_unit[k] (-1 none; a Consumed build that trains it), adjacent_terrain_rev[t]
    /// bit k = kind k needs an owned neighbour of terrain t, no_adjacent_terrain_bits; a resource-levelled kind's
    /// level_resource (0 none), max_level and growth_rate (Attract: one more such resource every growth_rate turns).
    pub creates_unit: [i32; 128],
    pub adjacent_terrain_rev: [u128; 16],
    pub no_adjacent_terrain_bits: u128,
    pub level_resource: [i32; 128],
    pub max_level: [i32; 128],
    pub growth_rate: [i32; 128],
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
    /// Polaris unit abilities: Skate, AutoFreeze, FreezeArea, Freeze.
    pub skate: [bool; 128],
    pub autofreeze: [bool; 128],
    pub freeze_area: [bool; 128],
    pub freeze: [bool; 128],
    /// Unit abilities read by ability rather than by kind: Heal ("HealOthers") and Convert (the Mind Bender and the
    /// Aquarion Siren).
    pub heal_others: [bool; 128],
    pub convert: [bool; 128],
    /// Aquarion unit abilities: Amphibious, Water ('s water-bound units without Carry),
    /// Tentacles, AutoFlood, Drench. Swim is `skills[FLOAT]`.
    pub amphibious: [bool; 128],
    pub water: [bool; 128],
    pub tentacles: [bool; 128],
    pub autoflood: [bool; 128],
    pub drench: [bool; 128],
    /// Elyrion: Fly (BabyDragon, FireDragon).
    pub fly: [bool; 128],
    /// Cymanti: Sneak (Hexapod, Moth): no zone of control on its moves.
    pub sneak: [bool; 128],
    /// Cymanti: the Poison ability (Kiton, Phychi, Exida, Moth, Island) and the Algae ability
    /// (the Island's trail, 2.2).
    pub poisons: [bool; 128],
    pub algae_trail: [bool; 128],
    /// Cymanti step 5: Explode (Doomux, Segment, bug egg, Boomchi), Swarm (Shaman), DoubleAttack (Phychi).
    pub explode: [bool; 128],
    pub swarm: [bool; 128],
    pub double_attack: [bool; 128],
    /// Cymanti step 6: Eat (the Centipede grows a Segment on a kill).
    pub eat: [bool; 128],
    /// Elyrion Grow: grows_into[k] (-1 none) after growth_rate[k] turns.
    pub grow: [bool; 128],
    pub grows_into: [i32; 128],
    pub growth_rate: [i32; 128],
    pub unit_tech: [i32; 128],
    pub structures: StructureRules,
    pub free_tech: [bool; 128],
    pub free_requires: [i32; 128],
    pub skills: [[bool; 10]; 128],
    pub engine_revision: String,
    /// Derived: tech_pred[k] bit i = tech_edges[i][k] (research reachability).
    pub tech_pred: [u128; 128],
    /// Derived: tech_closure[k] = k and every tech it requires, transitively (history.rs: a seen knight implies Chivalry,
    /// Free Spirit and Riding).
    pub tech_closure: [u128; 128],
    /// Derived: bits128(unit_supported), bits128(tech_exists).
    pub supported_bits: u128,
    pub tech_exists_bits: u128,
    /// Derived: tech_succ[i] bit k = tech_edges[i][k]; units_by_tech[i]: unit kinds tech i unlocks (cost >= 1,
    /// unit_tech < 127); unit_static: the Warrior (kind 2) when summonable without a tech; unit_cost_le.
    pub tech_succ: [u128; 128],
    pub units_by_tech: [u128; 128],
    pub unit_static: u128,
    pub unit_cost_le: CostIndex,
    /// get all tech for tribe: the techs reachable from Basic in breadth-first order over `next` (the
    /// game's tech unlocks, in its order), each with its `next` list. get unlockable tech walks it.
    pub tech_tree: Vec<(usize, Vec<usize>)>,
    /// Polaris, Aquarion and Elyrion: the same tables for a Polaris / Aquarion / Elyrion seat, through the tribe's
    /// tribe data overrides (tribes.rs). Every other tribe plays the regular tables.
    pub polaris: TribeTables,
    pub aquarion: TribeTables,
    pub elyrion: TribeTables,
    pub cymanti: TribeTables,
}

/// What a tribe's overrides change in the derived tables: research successors, get all tech for tribe, trainable units
/// by tech, improvements by tech, and the improvements the tribe may build at all.
#[derive(Clone)]
pub struct TribeTables {
    pub tech_succ: [u128; 128],
    pub tech_tree: Vec<(usize, Vec<usize>)>,
    pub units_by_tech: [u128; 128],
    pub structure_unlocks: [u128; 128],
    pub static_ok: u128,
    /// The techs of the tribe's tree: the regular ones with every overridden tech replaced by its override.
    pub tech_exists_bits: u128,
    /// The units summonable without a tech (the regular Warrior, through the unit override).
    pub unit_static: u128,
}

impl TribeTables {
    fn empty() -> TribeTables {
        TribeTables { tech_succ: [0; 128], tech_tree: Vec::new(), units_by_tech: [0; 128], structure_unlocks: [0; 128],
                      static_ok: 0, tech_exists_bits: 0, unit_static: 0 }
    }
}

/// A tribe's tables from the regular ones: every tech id through get override, the BFS of
/// get all tech for tribe over the overridden techs' own tech unlocks, units and improvements through their overrides.
fn tribe_tables(r: &Rules, next: &[Vec<usize>], structures: &[StructRow], tribe: i32) -> TribeTables {
    use crate::tribes::{structure_override, tech_override, unit_override};
    let map_bits = |b: u128, f: &dyn Fn(usize) -> usize| (0..128).filter(|&k| b >> k & 1 == 1).fold(0u128, |a, k| a | 1u128 << f(k));
    let mut t = TribeTables { tech_exists_bits: map_bits(r.tech_exists_bits, &|k| tech_override(tribe, k)),
                              unit_static: map_bits(r.unit_static, &|k| unit_override(tribe, k as i32) as usize),
                              ..TribeTables::empty() };
    if tribe == crate::tribes::POLARIS {
        // Sledding replaces Aquaculture (39), which the regular tree here does not have: Frostwork's child
        t.tech_exists_bits |= 1u128 << 41;
    }
    for i in 0..128 {
        t.tech_succ[i] = map_bits(r.tech_succ[i], &|k| tech_override(tribe, k));
        t.units_by_tech[i] = map_bits(r.units_by_tech[i], &|k| unit_override(tribe, k as i32) as usize);
        t.structure_unlocks[i] = map_bits(r.structures.tech_unlocks[i], &|k| structure_override(tribe, k));
    }
    // the tribe's own techs' unit unlocks (Aquarion: aqua base, owned from the start, is outside the tree and unlocks
    // no unit; get unlocked units reads every owned tech, tree or not)
    for &(tech, unit) in crate::tribes::tech_units(tribe) {
        t.units_by_tech[tech] |= 1u128 << unit;
    }
    let (mut queue, mut at) = (vec![0usize], 0usize);
    while at < queue.len() && at < 200 {
        let k = queue[at];
        at += 1;
        if t.tech_exists_bits >> k & 1 == 1 && !t.tech_tree.iter().any(|(e, _)| *e == k) {
            let kids: Vec<usize> = next[k].iter().map(|&c| tech_override(tribe, c)).collect();
            queue.extend(kids.iter().copied());
            t.tech_tree.push((k, kids));
        }
    }
    // get all tech for tribe is the tree: a regular tech only an overridden one leads to (Ramming, under Fishing) is not in
    // a Polaris game
    t.tech_exists_bits = t.tech_tree.iter().fold(0u128, |b, (k, _)| b | 1u128 << k);
    let rs = &r.structures;
    for st in structures {
        let k = st.id;
        let allowed = st.tribe.is_none() || st.tribe == Some(tribe as i64);
        t.static_ok |= ((allowed && rs.cost[k] >= 0 && rs.resource[k] == 0) as u128) << k;
    }
    t
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
    /// The override tables of a special tribe (Polaris, Aquarion, Elyrion, Cymanti); None for the regular tribes.
    #[inline]
    pub fn special(&self, tribe: i32) -> Option<&TribeTables> {
        match tribe {
            crate::tribes::POLARIS => Some(&self.polaris),
            crate::tribes::AQUARION => Some(&self.aquarion),
            crate::tribes::ELYRION => Some(&self.elyrion),
            crate::tribes::CYMANTI => Some(&self.cymanti),
            _ => None,
        }
    }
    /// The tech tree of a tribe: get all tech for tribe through its overrides.
    #[inline]
    pub fn tech_tree_of(&self, tribe: i32) -> &Vec<(usize, Vec<usize>)> {
        self.special(tribe).map_or(&self.tech_tree, |t| &t.tech_tree)
    }
    #[inline]
    pub fn tech_succ_of(&self, tribe: i32) -> &[u128; 128] {
        self.special(tribe).map_or(&self.tech_succ, |t| &t.tech_succ)
    }
    #[inline]
    pub fn units_by_tech_of(&self, tribe: i32) -> &[u128; 128] {
        self.special(tribe).map_or(&self.units_by_tech, |t| &t.units_by_tech)
    }
    #[inline]
    pub fn unit_static_of(&self, tribe: i32) -> u128 {
        self.special(tribe).map_or(self.unit_static, |t| t.unit_static)
    }
    #[inline]
    pub fn structure_unlocks_of(&self, tribe: i32) -> &[u128; 128] {
        self.special(tribe).map_or(&self.structures.tech_unlocks, |t| &t.structure_unlocks)
    }
    #[inline]
    pub fn tech_exists_of(&self, tribe: i32) -> u128 {
        self.special(tribe).map_or(self.tech_exists_bits, |t| t.tech_exists_bits)
    }
    #[inline]
    pub fn structure_static_ok_of(&self, tribe: i32) -> u128 {
        self.special(tribe).map_or(self.structures.static_ok, |t| t.static_ok)
    }
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
            skate: [false; 128],
            autofreeze: [false; 128],
            freeze_area: [false; 128],
            freeze: [false; 128],
            heal_others: [false; 128],
            convert: [false; 128],
            amphibious: [false; 128],
            water: [false; 128],
            tentacles: [false; 128],
            autoflood: [false; 128],
            drench: [false; 128],
            fly: [false; 128],
            sneak: [false; 128],
            poisons: [false; 128],
            algae_trail: [false; 128],
            explode: [false; 128],
            swarm: [false; 128],
            double_attack: [false; 128],
            eat: [false; 128],
            grow: [false; 128],
            grows_into: [-1; 128],
            growth_rate: [0; 128],
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
                creates_unit: [-1; 128],
                adjacent_terrain_rev: [0; 16],
                no_adjacent_terrain_bits: 0,
                level_resource: [0; 128],
                max_level: [0; 128],
                growth_rate: [0; 128],
            },
            free_tech: [false; 128],
            free_requires: [-1; 128],
            skills: [[false; 10]; 128],
            engine_revision: data.engine_revision.clone(),
            tech_pred: [0; 128],
            tech_closure: [0; 128],
            supported_bits: 0,
            tech_exists_bits: 0,
            tech_succ: [0; 128],
            units_by_tech: [0; 128],
            unit_static: 0,
            unit_cost_le: CostIndex { min: 0, table: vec![0] },
            tech_tree: Vec::new(),
            polaris: TribeTables::empty(),
            aquarion: TribeTables::empty(),
            elyrion: TribeTables::empty(),
            cymanti: TribeTables::empty(),
        };
        let core_names = [
            "Raft", "Rammership", "Scoutship", "Bomber", "Juggernaut", "MindBender", "Warrior", "Rider", "Knight",
            "Defender", "Catapult", "Archer", "Swordsman", "Giant", "Cloak", "Dagger", "Dinghy", "Pirate",
            // Polaris (a batch offers Polaris only with the opt-in, tribes:wip)
            "Mooni", "BattleSled", "IceFortress", "IceArcher", "Gaami",
            // Aquarion (opt-in as well)
            "MermaidWarrior", "Amphibian", "MermaidArcher", "MermaidDefender", "MermaidSwordsman", "Siren", "Tridention",
            "MermaidCloak", "MermaidDagger", "Shark", "Jelly", "Aquapult", "Crab",
            // Elyrion (opt-in as well): enchant animal's Polytaur, the super unit DragonEgg and what it grows into
            "Polytaur", "DragonEgg", "BabyDragon", "FireDragon",
            // Cymanti (opt-in as well): the start unit, the trainable overrides, the reward units and the Grow chain
            "Shaman", "Hexapod", "Doomux", "Phychi", "Kiton", "Exida", "Centipede", "Segment", "Raychi", "Boomchi",
            "LivingIsland", "Mantis", "InsectEgg", "Moth", "Larva",
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
            r.skate[i] = u.skills.iter().any(|s| s == "Skate");
            r.autofreeze[i] = u.skills.iter().any(|s| s == "AutoFreeze");
            r.freeze_area[i] = u.skills.iter().any(|s| s == "FreezeArea");
            r.freeze[i] = u.skills.iter().any(|s| s == "Freeze");
            r.heal_others[i] = u.skills.iter().any(|s| s == "HealOthers");
            r.convert[i] = u.skills.iter().any(|s| s == "Convert");
            r.amphibious[i] = u.skills.iter().any(|s| s == "Amphibious");
            r.water[i] = u.skills.iter().any(|s| s == "Water");
            r.tentacles[i] = u.skills.iter().any(|s| s == "Tentacles");
            r.autoflood[i] = u.skills.iter().any(|s| s == "AutoFlood");
            r.drench[i] = u.skills.iter().any(|s| s == "Drench");
            r.fly[i] = u.skills.iter().any(|s| s == "Fly");
            r.sneak[i] = u.skills.iter().any(|s| s == "Sneak");
            r.poisons[i] = u.skills.iter().any(|s| s == "Poison");
            r.algae_trail[i] = u.skills.iter().any(|s| s == "Algae");
            r.explode[i] = u.skills.iter().any(|s| s == "Explode");
            r.swarm[i] = u.skills.iter().any(|s| s == "Swarm");
            r.double_attack[i] = u.skills.iter().any(|s| s == "DoubleAttack");
            r.eat[i] = u.skills.iter().any(|s| s == "Eat");
            r.grow[i] = u.skills.iter().any(|s| s == "Grow");
            r.grows_into[i] = u.grows_into.map_or(-1, |k| k as i32);
            r.growth_rate[i] = u.growth_rate.unwrap_or(0);
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
            s.creates_unit[i] = st.creates_unit.map_or(-1, |k| k as i32);
            for &t in &st.adjacent_terrain {
                if t < 16 {
                    s.adjacent_terrain_rev[t] |= 1u128 << i;
                }
            }
            s.level_resource[i] = st.level_resource.unwrap_or(0);
            s.max_level[i] = st.max_level.unwrap_or(0);
            s.growth_rate[i] = st.growth_rate.unwrap_or(0);
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
            let needs_terrain = rs.adjacent_terrain_rev.iter().any(|&b| b >> k & 1 == 1);
            rs.no_adjacent_terrain_bits |= (!needs_terrain as u128) << k;
            rs.per_city_bits |= (rs.per_city[k] as u128) << k;
        }
        for k in 0..128 {
            r.tech_closure[k] = 1u128 << k;
        }
        loop {
            let mut changed = false;
            for k in 0..128 {
                let mut c = r.tech_closure[k];
                let mut pred = r.tech_pred[k];
                while pred != 0 {
                    let i = pred.trailing_zeros() as usize;
                    pred &= pred - 1;
                    c |= r.tech_closure[i];
                }
                if c != r.tech_closure[k] {
                    r.tech_closure[k] = c;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
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
        r.polaris = tribe_tables(&r, &next, &data.structures, crate::tribes::POLARIS);
        r.aquarion = tribe_tables(&r, &next, &data.structures, crate::tribes::AQUARION);
        r.elyrion = tribe_tables(&r, &next, &data.structures, crate::tribes::ELYRION);
        r.cymanti = tribe_tables(&r, &next, &data.structures, crate::tribes::CYMANTI);
        Ok(r)
    }
}
