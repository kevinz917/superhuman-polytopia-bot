//! gpu/capture.py: village/city capture, starfish and ruins (native profile).

use crate::diplomacy::protected;
use crate::discovery::{explorer, movable};
use crate::economy::{add_population, claim, discover};
use crate::geom::Geom;
use crate::hash::hash_range;
use crate::military::{promote, push, summon};
use crate::rules::{Rules, INDEPENDENT};
use crate::state::State;
use crate::structures::{adjust_building_population, capital_index, connections, population_contribution};

pub fn capture_mask(s: &State, out: &mut [bool]) {
    let u = &s.units;
    let cp = s.current_player;
    for i in 0..s.n {
        // can capture / can examine ruins: no leader (Cymanti)
        let idle = u.owner[i] == cp && !u.moved[i] && !u.attacked[i] && u.leader[i] == 0;
        out[i] = idle && capturable(s, i);
    }
}

/// capture_mask's tile condition (a foreign unprotected village/city, a ruin, or a starfish with Navigation).
#[inline]
pub fn capturable(s: &State, i: usize) -> bool {
    let t = &s.tiles;
    let cp = s.current_player;
    // Star Fishing is a build (can build), so not on a tile that has an improvement
    (t.structure[i] == 1 && t.owner[i] != cp && !protected(s, cp, t.owner[i])) || t.structure[i] == 2
        || (t.resource[i] == 8 && s.players.tech[s.p()][crate::tribes::starfishing_tech(s.players.tribe[s.p()])] && t.structure[i] == 0)
}

pub fn city(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let n = s.n;
    let before = population_contribution(s, r, g);
    let owner = s.current_player;
    let old = s.tiles.owner[source];
    let enemy = old > 0;
    let old_order = s.cities.order[source];
    let order = (0..n).filter(|&i| s.cities.owner[i] == owner).map(|i| s.cities.order[i]).max().unwrap_or(0).max(0) + 1;
    let area: Vec<bool> = (0..n).map(|t| g.d(source, t) <= 1).collect();
    // capture city action / get city area: a captured city explores the square of its border radius and rules
    // the tiles there that no other city rules
    let radius = s.cities.border[source].max(1);
    let city_area: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    let membership: Vec<bool> = if enemy {
        (0..n).map(|t| city_area[t] && (s.tiles.city[t] < 0 || s.tiles.city[t] == source as i32)).collect()
    } else {
        area.clone()
    };
    let was_connected = enemy && s.cities.connected[source];
    // capture city action: the city's improvement score (calculate improvement score) changes hands
    let transfer = crate::economy::city_score(s, source);
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
        } else {
            // The game owes a reward only through a pending city level up trigger (level up city),
            // which capture city action never adds: it keeps the
            // level and the rewards list. Rewards the old owner never chose (a lost trigger, an imported game) are
            // not the capturer's: kind 0 (no reward has it, as for Luxidoor's start levels) makes up the count, so
            // no reward is pending and the next level-up offers its own level's pair.
            let count: i32 = c.rewards[source * 9..source * 9 + 9].iter().sum();
            c.rewards[source * 9] += (c.level[source] - 1 - count).max(0);
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
                // wipe player action: every unit of the wiped player is disbanded, refund included
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
    discover(s, if enemy { &city_area } else { &area }, owner);
    // capture city action: after the explore actions, Polaris freezes the city area and any
    // other player restores the climate there
    crate::polaris::city_area_climate(s, source, owner);
    // capture city action: the city tile becomes Field; the Flooded tribe (Aquarion)
    // floods it, any other capturer drains it (no routes of its own: capture command's update routes below)
    s.tiles.terrain[source] = 3;
    s.tiles.flooded[source] = s.players.tribe[s.pclip(owner)] == crate::tribes::AQUARION;
    if enemy {
        // capture city action: capturing a player's city counts as an attack on them
        crate::diplomacy::hostile(s, old);
    }
    // capture command queues update city connections action for [the
    // capturer, the tile's previous owner] under update routes action and capture city action: once the capture has run,
    // the routes are updated, then both players' connections
    crate::structures::update_routes(s, r);
    let mut hub_loss = false;
    if was_connected {
        // update city connections for a city connected to its old owner's capital: the old owner's start
        // tile loses its +1; if the new owner's network reaches the city, their start tile gains it and the city keeps
        // its own (change city connection action), else the city loses its +1 (disconnect city action)
        let n = s.n;
        let (old_cap, old_holds) = capital_index(s, old);
        if old_holds {
            let mut v = vec![0i32; n];
            v[old_cap] = -1;
            crate::economy::add_population_vector(s, &v, old, true);
        }
        let (reached, cap, holds) = crate::structures::reachable(s, owner, g, true);
        if holds && reached[source] && source != cap {
            let mut v = vec![0i32; n];
            v[cap] = 1;
            crate::economy::add_population_vector(s, &v, owner, true);
        } else {
            // a recaptured start tile is the network's hub, not a connected city: it disconnects after the new
            // owner's cities reconnect to it (recorded games: the hub levels up on the way)
            s.cities.connected[source] = false;
            hub_loss = true;
        }
    }
    connections(s, owner, g, true);
    if hub_loss {
        let mut v = vec![0i32; s.n];
        v[source] = -1;
        crate::economy::add_population_vector(s, &v, owner, true);
    }
    if enemy {
        connections(s, old, g, true);
    }
    crate::military::make_exhausted(s, source);
    s.units.invisible[source] = false;
    for i in 0..n {
        if s.units.owner[i] != owner && s.units.home[i] == source as i32 + 1 {
            s.units.home[i] = 0;
        }
    }
    let k = s.units.kind[source] as usize;
    s.units.home[source] = if r.skills[k][INDEPENDENT] { 0 } else { source as i32 + 1 };
}

/// build action(star fishing): a Manual improvement, so the unit is only marked moved
/// and attacked (double ready and Boosted stay, unlike make exhausted).
pub fn starfish(s: &mut State, source: usize) {
    s.units.moved[source] = true;
    s.units.attacked[source] = true;
    s.units.move_lock[source] = false;
    s.units.invisible[source] = false;
    s.tiles.resource[source] = 0;
    let p = s.p();
    s.players.stars[p] += 8;
}

/// get unlockable tect default: walking get all tech for tribe, the unknown `next` techs of every known
/// tech, then any unknown tech with a known `next` tech, each once, in that order.
pub fn unlockable_tech(r: &Rules, tribe: i32, tech: &[bool; 128]) -> Vec<usize> {
    let mut out = Vec::new();
    for (k, next) in r.tech_tree_of(tribe) {
        if tech[*k] {
            for &c in next {
                if r.tech_exists_of(tribe) >> c & 1 == 1 && !tech[c] && !out.contains(&c) {
                    out.push(c);
                }
            }
        } else if next.iter().any(|&c| tech[c]) && !out.contains(k) {
            out.push(*k);
        }
    }
    out
}

// ruins reward
const RESOURCES: u32 = 1;
const POPULATION: u32 = 2;
const EXPLORER: u32 = 3;
const FREE_TECH: u32 = 4;
const BATTLESHIP: u32 = 6;
const SWORDSMAN: u32 = 8;
const CITY: u32 = 10;

/// get ruins reward default: [Resources, population growth, Swordsman, Battleship, Explorer, free tech]
/// shuffled by random hash range(i, count, x, y, i) (Fisher-Yates); the first valid entry wins, so the reward is a
/// fixed function of the tile, the game seed and the examining player's position.
pub fn ruin_reward(s: &State, source: usize, r: &Rules, g: &Geom) -> u32 {
    let p = s.p();
    let tech = &s.players.tech[p];
    let (x, y) = ((source % s.size) as u32, (source / s.size) as u32);
    // version >= 120: a Flooded-tribe (Aquarion) player on a water tile
    // that is a valid lost-city location gets a city
    if s.players.tribe[p] == crate::tribes::AQUARION && lost_city_site(s, source, g) {
        return CITY;
    }
    let mut list = [RESOURCES, POPULATION, SWORDSMAN, BATTLESHIP, EXPLORER, FREE_TECH];
    for i in 0..list.len() - 1 {
        let j = hash_range(s.initial_seed, i as u32, list.len() as u32, &[x, y, i as u32]) as usize;
        list.swap(i, j);
    }
    let water = s.tiles.terrain[source] == 1 || s.tiles.terrain[source] == 2;
    for reward in list {
        let valid = match reward {
            // a neighbour the player can enter (can be accessed by player) and an unexplored tile within 2
            EXPLORER => {
                let mut enter = false;
                g.for_ball(source, 1, |t| {
                    enter |= t != source && (movable(tech, s.tiles.terrain[t]) || s.tiles.structure[t] == 48)
                });
                let mut fog = false;
                g.for_ball(source, 2, |t| fog |= t != source && !s.tiles.explored[p][t]);
                enter && fog
            }
            FREE_TECH => !unlockable_tech(r, s.players.tribe[p], tech).is_empty(),
            // has unlocked movement on terrain(Ocean): Sailing (Aquarion: marine life)
            BATTLESHIP => water && crate::kernels::ocean_move(tech),
            // another living player: the game's nature player (id 255) always counts as one
            SWORDSMAN => !water,
            _ => true,
        };
        if valid {
            return reward;
        }
    }
    RESOURCES
}

/// valid lost city location: not on the map edge, an
/// unowned Ocean tile, no city (village) or ruin within distance 2 (the examined ruin itself aside).
pub fn lost_city_site(s: &State, source: usize, g: &Geom) -> bool {
    let size = s.size;
    let (x, y) = (source % size, source / size);
    if x == 0 || y == 0 || x + 1 >= size || y + 1 >= size || s.tiles.owner[source] != 0 || s.tiles.terrain[source] != 2 {
        return false;
    }
    let mut near = false;
    g.for_ball(source, 2, |t| near |= t != source && (s.tiles.structure[t] == 1 || s.tiles.structure[t] == 2));
    !near
}

/// get modifier(resource data type) of a continent of `tribe`:
/// world continent constructor's defaults (game, crop, fish, fruit 0.5, metal 0.8, whale, spores 0), set player's aquacrop
/// 0.3 (version 123), then the tribe's resource modifier (tribe data; a positive default is scaled, a zero one
/// replaced). The map generator's dry preset (wetness 0) also zeroes fish and whale; it has no ocean, so no lost city.
pub fn continent_resource_mod(tribe: i32, resource: usize) -> f32 {
    const BASE: [f32; 10] = [0.0, 0.5, 0.5, 0.5, 0.0, 0.8, 0.5, 0.0, 0.0, 0.3];
    let m: &[(usize, f32)] = match tribe {
        2 => &[(2, 0.1)],
        3 => &[(2, 0.5), (9, 1.5)],
        4 => &[(2, 0.0)],
        5 => &[(2, 1.5)],
        7 => &[(6, 2.0), (1, 0.5)],
        8 => &[(3, 1.5)],
        10 => &[(1, 0.2)],
        11 => &[(2, 0.1), (6, 2.0)],
        12 => &[(6, 0.1), (3, 0.1), (1, 0.1), (5, 2.0)],
        13 => &[(5, 1.5)],
        14 => &[(6, 1.5)],
        15 => &[(6, 0.5)],
        17 => &[(7, 0.3), (2, 0.0)],
        _ => &[],
    };
    let mut v = BASE[resource];
    for &(k, f) in m {
        if k == resource {
            v = if 0.0 < v { v * f } else { f };
        }
    }
    v
}

/// add resource's pick with random hash: over the ten resources, a
/// candidate has a non-zero continent modifier m, Value(x, y, i) <= m and its terrain (resource terrain requirements;
/// whale never: its modifier is 0); the pick is candidates[Range(0, count, x, y, 10)].
fn pick_resource(seed: u32, x: i32, y: i32, terrain: i32, continent: i32) -> i32 {
    let mut cand = [0i32; 10];
    let mut found = 0;
    for i in 0..10 {
        let m = continent_resource_mod(continent, i);
        let fits = match i {
            1 => terrain == 5,
            2 | 6 => terrain == 3,
            3 | 8 | 9 => terrain == 1 || terrain == 2,
            4 => terrain == 2,
            5 => terrain == 4,
            7 => terrain == 3 || terrain == 5,
            _ => false,
        };
        if m != 0.0 && crate::hash::hash_value3(seed, x, y, i as i32) <= m && fits {
            cand[found] = i as i32;
            found += 1;
        }
    }
    if found == 0 { 0 } else { cand[crate::hash::hash_range3(seed, 0, found as i32, x, y, 10) as usize] }
}

/// examine ruins action's City reward: the orthogonal non-Ice neighbours
/// become Water; the tile becomes a Flooded Field with a new city; add resource on every free neighbour
/// (Value(x, y) <= 1 always holds) and every free tile at distance 2 with Value(x, y) <= 1/3 (free: no resource,
/// improvement or owner), a tile without a continent taking the examining tribe's new one; then capture city action,
/// update routes and update city connections, which capture:city does for an unowned village.
fn lost_city(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let size = s.size as i32;
    let (x, y) = ((source % s.size) as i32, (source / s.size) as i32);
    for (dx, dy) in [(-1, 0), (0, -1), (0, 1), (1, 0)] {
        let (nx, ny) = (x + dx, y + dy);
        if nx >= 0 && ny >= 0 && nx < size && ny < size {
            let t = (ny * size + nx) as usize;
            if s.tiles.terrain[t] != 6 {
                s.tiles.terrain[t] = 1;
            }
        }
    }
    s.tiles.structure[source] = 1;
    s.tiles.terrain[source] = 3;
    s.tiles.flooded[source] = true;
    let seed = s.initial_seed;
    let tribe = s.players.tribe[s.p()];
    let mut area = Vec::with_capacity(24);
    g.for_ball(source, 2, |t| if t != source { area.push(t) });
    for t in area {
        let (tx, ty) = ((t % s.size) as i32, (t / s.size) as i32);
        let near = (tx - x).abs() <= 1 && (ty - y).abs() <= 1;
        let value = crate::hash::hash_words(seed, &[tx as u32, ty as u32]) as f32 * 2.3283064e-10f32;
        let free = s.tiles.resource[t] == 0 && s.tiles.structure[t] == 0 && s.tiles.owner[t] == 0;
        if value <= (if near { 1.0 } else { 0.33333334f32 }) && free {
            if s.tiles.continent[t] == 0 {
                s.tiles.continent[t] = tribe;
            }
            s.tiles.resource[t] = pick_resource(seed, tx, ty, s.tiles.terrain[t], s.tiles.continent[t]);
        }
    }
    city(s, source, r, g);
}

/// examine ruins command + examine ruins action.
pub fn ruin(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let p = s.p();
    let cp = s.current_player;
    let reward = ruin_reward(s, source, r, g);
    crate::military::make_exhausted(s, source);
    s.units.invisible[source] = false;
    s.tiles.structure[source] = 0;
    s.tiles.structure_level[source] = 0;
    s.tiles.structure_turn[source] = 0;
    crate::cov::hit(match reward { RESOURCES => 18, FREE_TECH => 19, POPULATION => 20, EXPLORER => 21, _ => 22 });
    let (x, y) = ((source % s.size) as u32, (source / s.size) as u32);
    match reward {
        CITY => lost_city(s, source, r, g),
        RESOURCES => s.players.stars[p] += 10,
        FREE_TECH => {
            // a research action at no cost for list[random hash range(0, count, x, y)]
            let list = unlockable_tech(r, s.players.tribe[p], &s.players.tech[p]);
            let kind = list[hash_range(s.initial_seed, 0, list.len() as u32, &[x, y]) as usize];
            s.players.tech[p][kind] = true;
            s.players.tech_turn[p][kind] = s.turn;
            s.players.score[p] = s.players.score[p].wrapping_add(100 * r.tech_tier[kind]);
        }
        POPULATION => {
            // three increase population actions on get current capital coordinates (the start tile, else the first own
            // city in tile order); nothing without a city
            if let Some(cap) = crate::structures::current_capital(s, cp) {
                add_population(s, cap, 3);
            }
        }
        EXPLORER => {
            let reveal = explorer(s, source, g);
            discover(s, &reveal, cp);
        }
        _ => {
            // train unit on occupied space (the examining unit is pushed off first), then promote action: a veteran unit 44
            // (our Rammership carrying a Warrior) on water, a veteran Swordsman on land; no home (not a city tile).
            // train unit on occupied space trains the tribe's override (Aquarion: a Shark,
            // which carries nothing, and a MermaidSwordsman)
            let water = reward == BATTLESHIP;
            let kind = crate::tribes::unit_override(s.players.tribe[p], if water { 44 } else { 11 });
            push(s, source, r, g);
            summon(s, source, kind as usize, r, g, false);
            promote(s, source, r);
            let carry = r.skills[kind as usize][crate::rules::CARRY];
            s.units.passenger[source] = if carry { 2 } else { 0 };
            s.units.home[source] = 0;
            // train unit gives a vehicle a Warrior passenger, and train action scores get unit score = 5 x get full cost,
            // passenger included
            if carry {
                s.players.score[p] = s.players.score[p].wrapping_add(5 * r.unit_cost[2]);
            }
        }
    }
}
