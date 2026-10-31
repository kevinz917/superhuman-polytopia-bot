//! gpu/infiltration.py: native Cloak infiltration.

use crate::geom::Geom;
use crate::hash::{hash_word, hash_words};
use crate::kernels::{combat, max_health};
use crate::military::{defense_bonus, remove, summon};
use crate::rules::Rules;
use crate::state::State;

/// Spawn tiles for `target`'s city in priority order (eligible only): the infiltration reward's tile
/// evaluation (city +2, water -0.1, known resource +0.5, defence bonus +1) with the reference's seeded tie noise.
pub fn spawn_order(s: &State, target: usize, g: &Geom) -> Vec<usize> {
    let (n, size) = (s.n, s.size);
    let t = &s.tiles;
    let tech = &s.players.tech[s.p()];
    let mut keyed: Vec<(f32, usize)> = Vec::new();
    for i in 0..n {
        let water = t.terrain[i] <= 2;
        let mut eligible = t.city[i] == target as i32 && s.units.owner[i] == 0;
        eligible &= (t.terrain[i] != 4 || tech[20]) && (t.terrain[i] != 1 || tech[10]) && (t.terrain[i] != 2 || tech[13]);
        // passability: only tiles the infiltrator's owner has explored
        eligible &= t.explored[s.p()][i];
        if !eligible {
            continue;
        }
        let defense = (t.terrain[i] == 4 && tech[20]) || (t.terrain[i] == 5 && tech[18]) || (water && tech[12]);
        let known_resource = t.resource[i] > 0 && (t.resource[i] != 2 || tech[6] || tech[8] || tech[9])
            && (t.resource[i] != 5 || tech[20] || tech[23] || tech[24]);
        let words = [(i % size) as u32, (i / size) as u32, s.turn as u32];
        let noise = (hash_word(s.initial_seed, hash_words(s.initial_seed, &words)) as f32) * 0.1f32 / 4294967296.0f32;
        let priority = noise + (t.structure[i] == 1) as i32 as f32 * 2.0 - water as i32 as f32 * 0.1 + known_resource as i32 as f32 * 0.5
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
    // infiltration reward: the city's raw production, no negative-progress term
    let innate = (c.production[target] + (s.tiles.capital[target] == c.owner[target]) as i32).max(0);
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
