//! gpu/economy.py: population, territory, harvest/forest actions, discovery and city rewards.

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

/// No unit blocks work on tile `t`: none, the mover's, a hidden one, or a unit of
/// a player the mover has peace with.
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
    // building: a `creates` terrain clears the resource (Consumed improvements destroy it), so Grow
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
