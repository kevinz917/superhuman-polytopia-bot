//! The tribes of a game's seats and what each gives at the start (game data v28 `tribeData`;
//! local game creation and capital placement as the project's full-game engine port has them).
//!
//! A seat's tribe travels with its map: the capitals array of a layout (engine-rs `reset_layouts` / `tick` layouts,
//! engine-cu `pc_reset` / `pc_tick` caps, the trainer's map tables) holds `tile | tribe << 16`. A plain tile index
//! (tribe bits 0) is Bardur, so every caller from before the tribes plays exactly what it played before.
//!
//! Increment R: the twelve regular tribes. The special tribes (Aquarion 3, Elyrion 5, Polaris 16, Cymanti 17) need
//! their own units, techs and mechanics and are refused until those are in.

pub const BARDUR: i32 = 4;
pub const LUXIDOOR: i32 = 9;
/// Bits of a packed capital that hold the tile.
pub const TILE_BITS: u32 = 16;

/// What a tribe starts with: its tribe tech (besides Basic), its start unit (unit type) and that unit's health,
/// its stars, and its capital's level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Start {
    pub tech: Option<usize>,
    pub unit: i32,
    pub health: f32,
    pub stars: i32,
    pub capital_level: i32,
}

/// game logic data 28 tribe data of a regular tribe: starting tech (after "basic"), starting unit, starting stars (5 when
/// absent), bonus CITY_LEVEL3 (Luxidoor: set tile as capital sets the capital's level to 3). Unit health is unit data
/// health / 10 (the engines keep hit points). None for the special tribes and anything else.
///
/// A capital that starts above level 1 (Luxidoor) had no city reward triggers for those levels; mapgen:initial_state
/// counts them in reward kind 0 (unused by every reward), so the engine's `reward_count < level - 1` pending test and
/// its level -> reward-pair choice stay the game's, and gives it engine-rs production = level (calculate work's
/// level + production - 1).
pub fn start(tribe: i32) -> Option<Start> {
    let s = |tech: Option<usize>, unit: i32, health: f32, stars: i32| Start { tech, unit, health, stars, capital_level: 1 };
    Some(match tribe {
        2 => s(Some(22), 10, 10.0, 5),   // Ai-Mo: Philosophy, MindBender
        4 => s(Some(15), 2, 10.0, 5),    // Bardur: Hunting, Warrior
        6 => s(Some(18), 9, 10.0, 7),    // Hoodrick: Archery, Archer
        7 => s(Some(6), 2, 10.0, 5),     // Imperius: Organization, Warrior
        8 => s(Some(10), 2, 10.0, 5),    // Kickoo: Fishing, Warrior
        9 => Start { tech: None, unit: 2, health: 10.0, stars: 2, capital_level: 3 },   // Luxidoor: Basic only
        10 => s(Some(1), 3, 10.0, 6),    // Oumaji: Riding, Rider
        11 => s(Some(7), 5, 15.0, 7),    // Quetzali: Strategy ("shields"), Defender
        12 => s(Some(24), 11, 15.0, 5),  // Vengir: Smithery, Swordsman
        13 => s(Some(20), 2, 10.0, 7),   // Xin-xi: Climbing, Warrior
        14 => s(Some(4), 2, 10.0, 7),    // Yadakk: Roads, Warrior
        15 => s(Some(8), 2, 10.0, 5),    // Zebasi: Farming, Warrior
        17 => s(Some(8), 36, 10.0, 5),   // Cymanti: Farming, Shaman (increment S4, work in progress: see `wip`)
        5 => s(Some(28), 2, 10.0, 5),    // Elyrion: forest magic, Warrior (increment S3, work in progress: see `wip`)
        3 => s(Some(45), 46, 10.0, 5),   // Aquarion: aqua base, MermaidWarrior (increment S2, work in progress: see `wip`)
        16 => s(Some(30), 22, 10.0, 5),  // Polaris: Frostwork, Mooni (increment S1, work in progress: see `wip`)
        _ => return None,
    })
}

/// Tier (tech data cost) of the start techs above, for the start score.
fn tier(tech: usize) -> i32 {
    match tech {
        1 | 6 | 10 | 15 | 20 | 28 | 30 => 1,
        4 | 7 | 8 | 18 => 2,
        22 | 24 => 3,
        _ => 0,
    }
}

/// unit data cost of the start units above.
fn unit_cost(unit: i32) -> i32 {
    match unit {
        2 | 46 => 2,
        3 | 5 | 9 => 3,
        10 | 11 | 22 | 36 => 5,
        _ => 0,
    }
}

/// The start score before exploration (mapgen:initial_state adds 5 per explored tile after): Generate's get score
/// pass (TILE_VALUE per owned tile, the capital's city score, 100 x cost per tech) plus create local game's
/// get unit score of the start unit (5 x cost). 390 for Bardur, as before; a level-3 capital scores 100 more.
pub fn start_score(st: &Start) -> i32 {
    390 - 100 * tier(15) - 5 * unit_cost(2) + st.tech.map_or(0, |t| 100 * tier(t)) + 5 * unit_cost(st.unit)
        + 50 * (st.capital_level - 1)
}

/// A tribe whose increment is still being built: the engine plays it, but a batch refuses it unless the caller opts
/// in with POLYTOPIA_GEN_WIP_TRIBES (a comma list of tribe ids), so nothing trains on a half-ported tribe.
pub fn wip(tribe: i32) -> bool {
    tribe == 16 || tribe == AQUARION || tribe == ELYRION || tribe == CYMANTI
}

/// The tribe is offered by `GameBatch`: in the engine and not work in progress (or opted in).
pub fn offered(tribe: i32) -> bool {
    start(tribe).is_some() && (!wip(tribe) || std::env::var("POLYTOPIA_GEN_WIP_TRIBES").map_or(false, |v| {
        v.split(',').any(|x| x.trim().parse::<i32>().ok() == Some(tribe))
    }))
}

/// (tile, tribe) of a packed capital; tribe bits 0 = Bardur.
pub fn unpack(cap: usize) -> (usize, i32) {
    let tribe = (cap >> TILE_BITS) as i32;
    (cap & ((1 << TILE_BITS) - 1), if tribe == 0 { BARDUR } else { tribe })
}

pub const POLARIS: i32 = 16;
pub const AQUARION: i32 = 3;
pub const ELYRION: i32 = 5;
pub const CYMANTI: i32 = 17;

/// game logic data 28 tribe data tech overrides (get override(tech data, tribe)). Polaris: Fishing -> Frostwork, Sailing ->
/// ice fishing, Navigation -> polar warfare, Aquatism -> Polarism, Aquaculture -> Sledding. Aquarion: Chivalry ->
/// Spearing, Sailing -> marine life, Ramming -> Aquaculture, Fishing -> aqua fishing, Roads -> Waterways. Elyrion:
/// Hunting -> forest magic, Whaling -> water magic (no Whaling: unreachable). Cymanti: Construction -> Recycling,
/// Chivalry -> Shock, Meditation -> Rituals, Philosophy -> Mysticism. Other tribes: none.
pub fn tech_override(tribe: i32, tech: usize) -> usize {
    match (tribe, tech) {
        (POLARIS, 10) => 30,
        (POLARIS, 13) => 42,
        (POLARIS, 14) => 31,
        (POLARIS, 12) => 32,
        (POLARIS, 39) => 41,
        (AQUARION, 3) => 26,
        (AQUARION, 13) => 46,
        (AQUARION, 44) => 39,
        (AQUARION, 10) => 47,
        (AQUARION, 4) => 48,
        (ELYRION, 15) => 28,
        (ELYRION, 11) => 29,
        (CYMANTI, 9) => 36,
        (CYMANTI, 3) => 35,
        (CYMANTI, 21) => 49,
        (CYMANTI, 22) => 50,
        (_, k) => k,
    }
}

/// tribe data unit overrides. Polaris: archer -> icearcher, giant -> gaami. Aquarion: giant -> crab, warrior ->
/// mermaidwarrior, archer -> mermaidarcher, swordsman -> mermaidswordsman, rider -> amphibian, cloak -> mermaidcloak,
/// dagger -> mermaiddagger, defender -> mermaiddefender, bombership -> jelly, rammership -> shark, mindbender -> siren,
/// catapult -> aquapult. Elyrion: giant -> dragonegg. Cymanti: rider -> hexapod, giant -> centipede, knight -> doomux,
/// archer -> phychi, swordsman -> mantis, catapult -> exida, defender -> kiton, mindbender -> shaman, cloak -> moth,
/// dagger and pirate -> bugegg, rammership -> boomchi, scoutship -> raychi, bombership -> island.
pub fn unit_override(tribe: i32, unit: i32) -> i32 {
    match (tribe, unit) {
        (POLARIS, 9) => 25,
        (POLARIS, 12) => 27,
        (AQUARION, 12) => 26,
        (AQUARION, 2) => 46,
        (AQUARION, 9) => 47,
        (AQUARION, 11) => 48,
        (AQUARION, 3) => 20,
        (AQUARION, 38) => 50,
        (AQUARION, 37) => 51,
        (AQUARION, 5) => 49,
        (AQUARION, 41) => 52,
        (AQUARION, 44) => 53,
        (AQUARION, 10) => 54,
        (AQUARION, 8) => 55,
        (ELYRION, 12) => 17,
        (CYMANTI, 3) => 28,
        (CYMANTI, 12) => 33,
        (CYMANTI, 4) => 29,
        (CYMANTI, 9) => 30,
        (CYMANTI, 11) => 59,
        (CYMANTI, 8) => 32,
        (CYMANTI, 5) => 31,
        (CYMANTI, 10) => 36,
        (CYMANTI, 38) => 61,
        (CYMANTI, 37) => 60,
        (CYMANTI, 40) => 60,
        (CYMANTI, 44) => 56,
        (CYMANTI, 42) => 35,
        (CYMANTI, 41) => 57,
        (_, u) => u,
    }
}

/// tribe data improvement overrides (Polaris: market -> icebank; Aquarion: burnforest -> fertilize; Elyrion: clearforest
/// -> sanctuary, burnforest -> none; Cymanti: market -> clathrus, farm -> fungi, port -> algae, road -> mycelium,
/// bridge -> none). clear forest and burn forest are forest-family slots, not build structures, in the engines: for
/// Elyrion the override shows as `forest_slot_ok` (slots 0 and 1 never legal). "None" maps to 0, the None structure,
/// which has no cost and so is never buildable (the bit drops out of every build row).
pub fn structure_override(tribe: i32, st: usize) -> usize {
    match (tribe, st) {
        (POLARIS, 50) => 34,
        (AQUARION, 11) => 53,
        (ELYRION, 10) => 32,
        (ELYRION, 11) => 0,
        (CYMANTI, 50) => 41,
        (CYMANTI, 5) => 37,
        (CYMANTI, 8) => 38,
        (CYMANTI, 3) => 39,
        (CYMANTI, 48) => 0,
        (_, k) => k,
    }
}

/// can player embark: a tech of the player unlocks an Embark improvement, the
/// Port through Fishing (10); Cymanti's port is the Algae, so never.
#[inline]
pub fn can_embark(tribe: i32, tech: &[bool; 128]) -> bool {
    tech[10] && tribe != CYMANTI
}

/// get tech price's Literacy (player ability literacy): Philosophy, or Mysticism for Cymanti.
#[inline]
pub fn literacy(tech: &[bool; 128]) -> bool {
    tech[22] || tech[50]
}

/// The tile's resource building and the tech that unlocks it (get improvements from tech through the improvement
/// overrides): Farm on crop (Farming; Cymanti's farm is the Fungi, so never), Mine on metal (Mining), Fungi on Spores
/// (Farming, Cymanti only), enchant animal on game (forest magic, which only Elyrion has) and Aquafarm on aquacrop
/// (Aquaculture). (0, 0): none.
#[inline]
pub fn resource_build(tribe: i32, resource: i32) -> (usize, usize) {
    match resource {
        2 if tribe != CYMANTI => (8, 5),
        5 => (23, 21),
        7 if tribe == CYMANTI => (8, 37),
        1 => (28, 30),
        9 => (39, 49),
        _ => (0, 0),
    }
}

/// Forest family slot `k` (0 burn forest, 1 clear forest, 2 grow forest) exists for the tribe: Elyrion's
/// improvement overrides turn clear forest into Sanctuary (a build-family structure) and burn forest into None
/// so slots 0 and 1 are never legal for an Elyrion seat.
#[inline]
pub fn forest_slot_ok(tribe: i32, k: usize) -> bool {
    !(tribe == ELYRION && k < 2)
}

/// tech data unit unlocks of the Polaris techs (Frostwork: Mooni, Sledding: BattleSled, polar warfare: IceFortress).
pub const POLARIS_TECH_UNITS: [(usize, i32); 3] = [(30, 22), (41, 23), (31, 24)];

/// tech data unit unlocks of the Aquarion techs (Spearing: Tridention, marine life: Shark).
pub const AQUARION_TECH_UNITS: [(usize, i32); 2] = [(26, 21), (46, 53)];

/// tech data unit unlocks of the Cymanti techs (Rituals: Shaman, Shock: Doomux).
pub const CYMANTI_TECH_UNITS: [(usize, i32); 2] = [(49, 36), (35, 29)];

/// The (tech, unit) unit unlocks a special tribe's own techs add (none for the regular tribes).
pub fn tech_units(tribe: i32) -> &'static [(usize, i32)] {
    match tribe {
        POLARIS => &POLARIS_TECH_UNITS,
        AQUARION => &AQUARION_TECH_UNITS,
        CYMANTI => &CYMANTI_TECH_UNITS,
        _ => &[],
    }
}

/// The tech that unlocks the Fishing improvement (Polaris: ice fishing, Aquarion: aqua fishing; tech data
/// improvement unlocks).
#[inline]
pub fn fishing_tech(tribe: i32) -> usize {
    match tribe {
        POLARIS => 42,
        AQUARION => 47,
        _ => 10,
    }
}

/// The tech that unlocks Star Fishing (Polaris: polar warfare) and its parent (starfish show once it is unlockable;
/// Aquarion: Navigation under marine life).
#[inline]
pub fn starfishing_tech(tribe: i32) -> usize {
    if tribe == POLARIS { 31 } else { 14 }
}

#[inline]
pub fn starfishing_parent(tribe: i32) -> usize {
    match tribe {
        POLARIS => 41,
        AQUARION => 46,
        _ => 13,
    }
}
