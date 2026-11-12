//! gpu/infiltration.py: native Cloak infiltration.

use crate::geom::Geom;
use crate::hash::{hash_word, hash_words};
use crate::kernels::{combat, max_health};
use crate::military::{defense_bonus, direction, direction_xy, remove, summon};
use crate::rules::Rules;
use crate::state::State;

/// Spawn tiles for `target`'s city in priority order (eligible only): infiltration reward action's
/// Evaluate: tie noise + city +2, water -0.1, a resource the player sees +0.5,
/// defence bonus +1, a Poison improvement -0.5 (not for a poison resist tribe: Cymanti). Sorted best first.
/// `bug`: the Poison reward (a Moth, Cymanti) spawns bug eggs and no Pirate, so only
/// the tiles a land unit can stand on count (water only with a Bridge or Algae).
pub fn spawn_order(s: &State, target: usize, g: &Geom, bug: bool) -> Vec<usize> {
    let (n, size) = (s.n, s.size);
    let t = &s.tiles;
    let tech = &s.players.tech[s.p()];
    let tribe = s.players.tribe[s.p()];
    let mut keyed: Vec<(f32, usize)> = Vec::new();
    for i in 0..n {
        let water = t.terrain[i] <= 2;
        let mut eligible = t.city[i] == target as i32 && s.units.owner[i] == 0;
        // is tile accessible: a Bridge (improvement ability BRIDGE) lets the Dagger stand on its water tile
        // whatever the techs; other water needs the Pirate's techs
        let bridge = t.structure[i] == 48;
        eligible &= (t.terrain[i] != 4 || tech[20]) && (bridge || if bug {
            !water || t.algae[i]
        } else {
            (t.terrain[i] != 1 || crate::kernels::water_move(tech)) && (t.terrain[i] != 2 || tech[13] || tech[46])
        });
        // is tile accessible: only tiles the infiltrator's owner has explored
        eligible &= t.explored[s.p()][i];
        if !eligible {
            continue;
        }
        let defense = (t.terrain[i] == 4 && tech[20]) || (t.terrain[i] == 5 && tech[18]) || (water && tech[12]);
        // get resource: no resource on a tile with an improvement (a farm, a mine, the
        // city), whatever lies under it
        // is resource visible to player (cymanti:resource_visible: Starfish once Star Fishing is unlockable, aqua crop only
        // with Aquaculture, Crop never and Spores only for Cymanti)
        let known_resource = t.structure[i] == 0 && crate::cymanti::resource_visible(tribe, tech, t.resource[i]);
        // an improvement with improvement ability Poison (Fungi 37, Clathrus 41) for a tribe without poison resist
        // (Bardur): -0.5
        let poison = (t.structure[i] == 37 || t.structure[i] == 41) && tribe != crate::tribes::CYMANTI;
        // tie noise: the game's random hash of (x, y, current turn), an integer hash over the three
        // values in that order, mapped to the range [0, 0.1):
        // 0.1 * XXHash(int) * 2^-32.
        // Our `turn` is the game's current turn.
        let words = [(i % size) as u32, (i / size) as u32, s.turn as u32];
        let noise = (hash_word(s.initial_seed, hash_words(s.initial_seed, &words)) as f32) * 0.1f32 / 4294967296.0f32;
        let priority = noise + (t.structure[i] == 1) as i32 as f32 * 2.0 - water as i32 as f32 * 0.1 + known_resource as i32 as f32 * 0.5
            + defense as i32 as f32 - poison as i32 as f32 * 0.5;
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
    // infiltration reward action: calculate raw production, no negative-progress term
    let innate = (c.production[target] + (s.tiles.capital[target] == c.owner[target]) as i32).max(0);
    let count = c.level[target].min(5) as usize;
    // infiltrate action: a Moth brings the Poison reward (Cymanti): no
    // currency, the stand-in Moth's battle, a surviving defender Poisoned, bug eggs spawned
    let moth = s.units.kind[source] == 61;
    let fake = if moth { 61 } else { 37 };
    remove(s, source, r, None, true);
    let enemy = s.units.owner[target] > 0 && s.units.owner[target] != owner;
    let tk = s.units.kind[target] as usize;
    let damage = combat(r.unit_attack[fake], r.unit_health[fake], r.unit_health[fake], r.unit_defense[tk], s.units.health[target],
                        max_health(s, r, target), defense_bonus(s, r, target)).0;
    s.units.health[target] += if enemy { -damage } else { 0.0 };
    // perform attack on the city's unit takes its Bubble (Aquarion)
    if enemy {
        s.units.bubble[target] = false;
        s.units.swift[target] = false;
    }
    let dead = enemy && s.units.health[target] <= 0.0;
    if dead {
        remove(s, target, r, None, true);
    } else if enemy {
        // perform attack(state, player, world coordinates null, city, damage): no unit on the
        // null origin (-1, -1), and the city's unit turns toward it (perform attack default)
        let size = s.size as i32;
        s.units.direction[target] = direction_xy((target as i32 % size, target as i32 / size), (-1, -1));
        if moth {
            s.units.poison[target] = true;
        }
    }
    s.players.kills[po] += dead as i32;
    if !moth {
        s.players.stars[po] += innate;
    }
    s.cities.infiltrated[target] = s.turn * s.np as i32 + owner;
    s.cities.riot[target] = true;
    let order = spawn_order(s, target, g, moth);
    // infiltration reward action: the spawn is the tribe's override of the Dagger (Aquarion: the MermaidDagger, a Swim
    // unit that stands on water itself)
    let dagger = crate::tribes::unit_override(s.players.tribe[po], if moth { 60 } else { 37 }) as usize;
    for (i, &at) in order.iter().enumerate() {
        if i >= count.min(5) {
            break;
        }
        // a Pirate (embark action first) only on water the Dagger cannot stand on: not on a Bridge
        let water = !moth && s.tiles.terrain[at] <= 2 && s.tiles.structure[at] != 48 && !r.skills[dagger][crate::rules::FLOAT];
        summon(s, at, if water { 40 } else { dagger }, r, g, false);
        s.units.home[at] = 0;
        // train action(player, Dagger, tile, 0, look coordinates = the city): train action turns
        // the new unit toward look coordinates (the same tile gives 8, none); on water the
        // queued embark action copies the facing to the Pirate
        s.units.direction[at] = direction(s.size, at, target);
    }
}
