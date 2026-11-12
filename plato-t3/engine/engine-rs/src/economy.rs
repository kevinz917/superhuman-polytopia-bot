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

/// calculate improvement score of city `i`: 50 per level + 50, 5 per point of progress (xp), the Park's
/// 250; with negative progress (version > 114) 100 more per point, floored at 0.
pub fn city_score(s: &State, i: usize) -> i32 {
    let c = &s.cities;
    let v = 5 * c.progress[i] + 50 * (c.level[i] + 1) + 250 * s.reward(i, 2);
    if c.progress[i] < 0 { (v + 100 * c.progress[i]).max(0) } else { v }
}

/// Give city `i` its place in the reward order when its reward has just become pending (State::reward_waits).
pub fn stamp_reward(s: &mut State, i: usize, was_pending: bool) {
    if was_pending || !s.reward_pending(i) {
        return;
    }
    let newest = (0..s.n).filter(|&j| j != i && s.reward_pending(j)).map(|j| s.cities.reward_seq[j]).max().unwrap_or(0);
    s.cities.reward_seq[i] = newest + 1;
}

/// level up while should level up (city level up action): progress pays for the next level.
pub fn level_up(s: &mut State, i: usize) {
    let was_pending = s.reward_pending(i);
    level_up_only(s, i);
    stamp_reward(s, i, was_pending);
}

fn level_up_only(s: &mut State, i: usize) {
    let c = &mut s.cities;
    while c.progress[i] >= c.level[i] + 1 {
        c.progress[i] -= c.level[i] + 1;
        c.level[i] += 1;
        c.production[i] += 1;
        if s.tiles.structure[i] > 0 {
            s.tiles.structure_level[i] += 1;
        }
    }
}

/// increase population action per city of `owner`: progress and score (the change of city_score). A city levels up
/// only on its owner's turn; otherwise at the owner's next turn start (start turn action, start_level_ups). Cities
/// never lose levels (negative progress stays).
pub fn add_population_vector(s: &mut State, amount: &[i32], owner: i32, _native: bool) {
    let n = s.n;
    let mut score: i32 = 0;
    for i in 0..n {
        if s.cities.owner[i] != owner || amount[i] == 0 {
            continue;
        }
        let before = city_score(s, i);
        s.cities.progress[i] += amount[i];
        s.cities.population[i] += amount[i];
        if owner == s.current_player {
            level_up(s, i);
        }
        score += city_score(s, i) - before;
    }
    let o = (owner - 1) as usize;
    s.players.score[o] = s.players.score[o].wrapping_add(score);
}

/// start turn action: the mover's cities that gained enough progress off-turn level up now (after
/// the turn's income, which reads the old level).
pub fn start_level_ups(s: &mut State, owner: i32) {
    let o = (owner - 1) as usize;
    // start turn action pushes a city level up action per city in tile order: they run from the highest tile down
    for i in (0..s.n).rev() {
        if s.cities.owner[i] == owner && s.cities.progress[i] >= s.cities.level[i] + 1 {
            let before = city_score(s, i);
            level_up(s, i);
            s.players.score[o] = s.players.score[o].wrapping_add(city_score(s, i) - before);
        }
    }
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

/// No unit blocks work on tile `t`: none, the mover's, a hidden one, or (can build /
/// can destroy) a unit of a player the mover has peace with.
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
    let fish = crate::tribes::fishing_tech(s.players.tribe[s.p()]);
    for t in 0..n {
        let r = s.tiles.resource[t];
        let unlocked = (r == 1 && tech[15]) || (r == 3 && tech[fish]) || (r == 6 && tech[6]);
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
    let tribe = s.players.tribe[s.p()];
    let aquarion = tribe == crate::tribes::AQUARION;
    let (burn_ok, clear_ok) = (crate::tribes::forest_slot_ok(tribe, 0), crate::tribes::forest_slot_ok(tribe, 1));
    for t in 0..n {
        let free = econ[t] && (s.tiles.structure[t] == 0 || s.tiles.structure[t] == 3);
        let forest = s.tiles.terrain[t] == 5;
        let field = s.tiles.terrain[t] == 3;
        out[t * 3] = free && burn_or_fertilize(s, t, aquarion, tech, stars) && burn_ok;
        out[t * 3 + 1] = free && forest && tech[16] && clear_ok;
        out[t * 3 + 2] = free && field && tech[19] && stars >= 5;
    }
}

/// Forest-family entry 0: Burn Forest (Construction, 3 stars, a forest), or for Aquarion its improvement override
/// Fertilize (fertilize: Construction, 5 stars, terrain wetland, creates crop). meets requirement
/// wetland is a flooded non-forest tile; can build
/// refuses a tile that already has the created crop.
#[inline]
pub fn burn_or_fertilize(s: &State, t: usize, aquarion: bool, tech: &[bool; 128], stars: i32) -> bool {
    if aquarion {
        s.tiles.flooded[t] && s.tiles.terrain[t] != 5 && s.tiles.resource[t] != 2 && tech[9] && stars >= 5
    } else {
        s.tiles.terrain[t] == 5 && tech[9] && stars >= 3
    }
}

/// Fertilize (build action): Consumed destroys the tile's resource, and
/// the crop it creates is added only if the tile had no resource the player sees (game logic data
/// .is resource visible to player: the resource's improvement unlocked or unlockable). Wetland is a Flooded Field (or a
/// Lost City's flooded neighbour turned Water); of the resources there only Starfish can be hidden from its owner
/// (Starfishing: Navigation, or marine life known), the others' improvements hang off root techs. So: no resource or a
/// hidden one -> crop, a visible one -> none.
pub fn fertilize(s: &mut State, target: usize) {
    let p = s.p();
    let res = s.tiles.resource[target];
    let tech = &s.players.tech[p];
    let visible = res != 0 && (res != 8 || tech[14] || tech[crate::tribes::starfishing_parent(s.players.tribe[p])]);
    s.tiles.resource[target] = if visible { 0 } else { 2 };
    s.players.stars[p] -= 5;
}

pub fn forest(s: &mut State, target: usize, ability: i32) {
    if ability == 1 && s.players.tribe[s.p()] == crate::tribes::AQUARION {
        return fertilize(s, target);
    }
    let terrain = if ability == 3 { 5 } else { 3 };
    // build action: a `creates` terrain clears the resource (Consumed improvements destroy it), so Grow
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
    // on reveal lighthouse tile: +1 to get current capital coordinates, nothing without a city
    if let Some(index) = crate::structures::current_capital(s, player) {
        add_population(s, index, count);
    }
}

/// out[t*9 + k]
pub fn reward_mask(s: &State, out: &mut [bool]) {
    let n = s.n;
    for t in 0..n {
        let count = s.reward_count(t);
        let pending = s.cities.owner[t] == s.current_player && count < s.cities.level[t] - 1 && !s.reward_waits(t);
        // the game stops at each level-up's reward trigger before the next population point (recorded games:
        // a city that gains two levels is offered the next level's pair, then the one after)
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
    // the city keeps its place in the reward order for its next pending level (the stack resumes with what this
    // reward pushes: a Population Growth level-up is offered next)
    let seq = s.cities.reward_seq[target];
    s.cities.rewards[target * 9 + kind as usize] += 1;
    s.cities.reward_seq[target] = 0;
    reward_effect(s, target, kind, r, g, n);
    s.cities.reward_seq[target] = if s.reward_pending(target) { if seq > 0 { seq } else { s.cities.reward_seq[target] } } else { 0 };
}

fn reward_effect(s: &mut State, target: usize, kind: i32, r: &Rules, g: &Geom, n: usize) {
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
            let cp = s.current_player;
            let polaris = s.players.tribe[s.p()] == crate::tribes::POLARIS;
            // expand city action: after the border's explore actions, Polaris freezes the city area
            // (Spread, tribe 16) and any other player restores the climate there
            crate::polaris::city_area_climate(s, target, cp);
            // city reward action (border growth): expand city action, update routes action, then
            // update city connections action for every other player (the grower's own come with the deferred update)
            crate::structures::update_routes(s, r);
            for q in (1..=s.np as i32).filter(|&q| q != cp) {
                crate::structures::connections(s, q, g, true);
            }
            if polaris {
                // add border growth actions, tribe 16: freeze area action(city, new border, no units, owned tiles only)
                let radius = s.cities.border[target];
                crate::polaris::freeze_area(s, target, radius, false, true, r, g);
            }
        }
        4 => {
            let reveal = crate::discovery::explorer(s, target, g);
            discover(s, &reveal, s.current_player);
        }
        6 => crate::military::superunit(s, target, r, g),
        _ => {}
    }
}
