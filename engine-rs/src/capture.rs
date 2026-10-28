//! gpu/capture.py: village/city capture, starfish and ruins (native profile).

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
