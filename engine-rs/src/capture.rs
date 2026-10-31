//! gpu/capture.py: village/city capture, starfish and ruins (native profile).

use crate::diplomacy::protected;
use crate::discovery::{explorer, movable};
use crate::economy::{add_population, claim, discover};
use crate::geom::Geom;
use crate::hash::hash_range;
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
    // Star Fishing is a build, so not on a tile that has an improvement
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
    let order = (0..n).filter(|&i| s.cities.owner[i] == owner).map(|i| s.cities.order[i]).max().unwrap_or(0).max(0) + 1;
    let area: Vec<bool> = (0..n).map(|t| g.d(source, t) <= 1).collect();
    // capturing a city: the captured city explores the square of its border radius and rules
    // the tiles there that no other city rules
    let radius = s.cities.border[source].max(1);
    let city_area: Vec<bool> = (0..n).map(|t| g.d(source, t) <= radius).collect();
    let membership: Vec<bool> = if enemy {
        (0..n).map(|t| city_area[t] && (s.tiles.city[t] < 0 || s.tiles.city[t] == source as i32)).collect()
    } else {
        area.clone()
    };
    let was_connected = enemy && s.cities.connected[source];
    // capturing a city: the city's improvement score changes hands
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
                // wiping a player out: every unit of the wiped player is disbanded, refund included
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
    if enemy {
        // capturing a player's city counts as an attack on them
        crate::diplomacy::hostile(s);
    }
    let mut hub_loss = false;
    if was_connected {
        // city connections, for a city connected to its old owner's capital: the old owner's start
        // tile loses its +1; if the new owner's network reaches the city, their start tile gains it and the city keeps
        // its own (it moves to the new network), else the city loses its +1 (it disconnects)
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

/// The techs a player can unlock: walking the tribe's tech list, the unknown `next` techs of every known
/// tech, then any unknown tech with a known `next` tech, each once, in that order.
pub fn unlockable_tech(r: &Rules, tech: &[bool; 128]) -> Vec<usize> {
    let mut out = Vec::new();
    for (k, next) in &r.tech_tree {
        if tech[*k] {
            for &c in next {
                if r.tech_exists[c] && !tech[c] && !out.contains(&c) {
                    out.push(c);
                }
            }
        } else if next.iter().any(|&c| tech[c]) && !out.contains(k) {
            out.push(*k);
        }
    }
    out
}

// ruins rewards
const RESOURCES: u32 = 1;
const POPULATION: u32 = 2;
const EXPLORER: u32 = 3;
const FREE_TECH: u32 = 4;
const BATTLESHIP: u32 = 6;
const SWORDSMAN: u32 = 8;

/// The ruins reward: [resources, population, Swordsman, Battleship, Explorer, free tech] shuffled by the
/// seeded hash of (x, y, i) modulo (count - i), plus i (Fisher-Yates); the first valid entry wins, so the reward is a
/// fixed function of the tile, the game seed and the examining player's position.
pub fn ruin_reward(s: &State, source: usize, r: &Rules, g: &Geom) -> u32 {
    let p = s.p();
    let tech = &s.players.tech[p];
    let (x, y) = ((source % s.size) as u32, (source / s.size) as u32);
    let mut list = [RESOURCES, POPULATION, SWORDSMAN, BATTLESHIP, EXPLORER, FREE_TECH];
    for i in 0..list.len() - 1 {
        let j = hash_range(s.initial_seed, i as u32, list.len() as u32, &[x, y, i as u32]) as usize;
        list.swap(i, j);
    }
    let water = s.tiles.terrain[source] == 1 || s.tiles.terrain[source] == 2;
    for reward in list {
        let valid = match reward {
            // a neighbour the player can enter and an unexplored tile within 2
            EXPLORER => {
                let mut enter = false;
                g.for_ball(source, 1, |t| {
                    enter |= t != source && (movable(tech, s.tiles.terrain[t]) || s.tiles.structure[t] == 48)
                });
                let mut fog = false;
                g.for_ball(source, 2, |t| fog |= t != source && !s.tiles.explored[p][t]);
                enter && fog
            }
            FREE_TECH => !unlockable_tech(r, tech).is_empty(),
            // ocean movement: Sailing
            BATTLESHIP => water && tech[13],
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

/// Examining ruins.
pub fn ruin(s: &mut State, source: usize, r: &Rules, g: &Geom) {
    let p = s.p();
    let cp = s.current_player;
    let reward = ruin_reward(s, source, r, g);
    exhaust(s, source);
    s.units.invisible[source] = false;
    s.tiles.structure[source] = 0;
    s.tiles.structure_level[source] = 0;
    s.tiles.structure_turn[source] = 0;
    crate::cov::hit(match reward { RESOURCES => 18, FREE_TECH => 19, POPULATION => 20, EXPLORER => 21, _ => 22 });
    let (x, y) = ((source % s.size) as u32, (source / s.size) as u32);
    match reward {
        RESOURCES => s.players.stars[p] += 10,
        FREE_TECH => {
            // the tech list[hash(x, y) modulo count], researched at no cost
            let list = unlockable_tech(r, &s.players.tech[p]);
            let kind = list[hash_range(s.initial_seed, 0, list.len() as u32, &[x, y]) as usize];
            s.players.tech[p][kind] = true;
            s.players.tech_turn[p][kind] = s.turn;
            s.players.score[p] = s.players.score[p].wrapping_add(100 * r.tech_tier[kind]);
        }
        POPULATION => {
            // three population points to the current capital (the start tile, else the first own
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
            // trained on an occupied tile (the examining unit is pushed off first), then promoted: a veteran unit 44
            // (our Rammership carrying a Warrior) on water, a veteran Swordsman on land; no home (not a city tile)
            let water = reward == BATTLESHIP;
            push(s, source, r, g);
            summon(s, source, if water { 44 } else { 11 }, r, g, false);
            promote(s, source, r);
            s.units.passenger[source] = if water { 2 } else { 0 };
            s.units.home[source] = 0;
        }
    }
}
