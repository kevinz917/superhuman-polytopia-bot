//! Aquarion (increment S2) step 4: flooding and the Bubble effect. Rules as in the
//! project's full-game engine port.
//!
//! Representation: `tiles.flooded` is tile effect flooded, `units.bubble` unit effect bubble. Skins are 0 in our games,
//! so Swamped (flood tile's and Drain's skin-17 half) never exists. The Bubble player ability is Waterways' (48)
//! ability unlocks; the Flooded tribe ability is Aquarion's alone (tribe data).
use crate::geom::Geom;
use crate::rules::{Rules, FLOAT};
use crate::state::State;

/// Waterways: Canal (its improvement unlocks) and the Bubble player ability (its ability unlocks).
pub const WATERWAYS: usize = 48;

/// is floodable: not water, not already Flooded, not Mountain or Ice,
/// i.e. a Field or Forest that is not Flooded.
#[inline]
pub fn floodable(s: &State, t: usize) -> bool {
    let k = s.tiles.terrain[t];
    !(k == 1 || k == 2) && !s.tiles.flooded[t] && (k & !2) != 4
}

/// player state has the Bubble ability: Waterways known.
#[inline]
pub fn bubble_ability(s: &State, player: i32) -> bool {
    player > 0 && s.players.tech[s.pclip(player)][WATERWAYS]
}

/// check bubble step as its bubble action runs: on a Flooded tile an
/// Amphibious unit without Bubble gains it.
#[inline]
pub fn check_bubble_step(s: &mut State, t: usize, r: &Rules) {
    let kind = s.units.kind[t].clamp(0, 127) as usize;
    if s.tiles.flooded[t] && s.units.owner[t] != 0 && r.amphibious[kind] {
        s.units.bubble[t] = true;
    }
}

/// flood tile action for `player`: skipped unless floodable; the tile is
/// Flooded; update routes then update city connections(player); a unit on the tile (any owner) is bubbled when the
/// flooding player has the Bubble ability (its bubble action runs after the two route actions; they commute).
pub fn flood_tile(s: &mut State, t: usize, player: i32, r: &Rules, g: &Geom) {
    if !floodable(s, t) {
        return;
    }
    s.tiles.flooded[t] = true;
    crate::structures::update_routes(s, r);
    crate::structures::connections(s, player, g, true);
    if s.units.owner[t] != 0 && bubble_ability(s, player) {
        check_bubble_step(s, t, r);
    }
}

/// move action's Bubble loss and AutoFlood for the unit that arrived on `t`,
/// `had` its Bubble before the move: an amphibious unit that had Bubble loses it on land that is not wetland; a Crab's
/// flood tile (not on a push) runs after that removal and may bubble it again. Returns whether `t` was Flooded when
/// the unit arrived (for bubble_on_arrival).
pub fn after_move(s: &mut State, t: usize, had: bool, mover: i32, r: &Rules, g: &Geom, autoflood: bool) -> bool {
    let kind = s.units.kind[t].clamp(0, 127) as usize;
    let flooded_before = s.tiles.flooded[t];
    let terrain = s.tiles.terrain[t];
    // unit data is_amphibious: Swim, no movement terrain, no Carry (equal to the Amphibious ability over)
    if had && (3..=6).contains(&terrain) && !flooded_before && r.amphibious[kind] {
        s.units.bubble[t] = false;
    }
    if autoflood && r.autoflood[kind] {
        flood_tile(s, t, mover, r, g);
    }
    flooded_before
}

/// move action's check bubble step, decided when the unit arrived and run last: a mover
/// with the Bubble ability bubbles an amphibious unit without Bubble on a tile that was Flooded.
pub fn bubble_on_arrival(s: &mut State, t: usize, had: bool, flooded_at_move: bool, mover: i32, r: &Rules) {
    if flooded_at_move && !had && s.tiles.flooded[t] && bubble_ability(s, mover) {
        check_bubble_step(s, t, r);
    }
}

/// move action's Tentacle trap for the unit that just arrived on
/// `t`: the destination has the Tentacle effect (derived: update tentacles runs after every change of a non-Frozen
/// Tentacles unit) and get tentacle owner finds, among its neighbours in
/// get tile neighbors order, the healthiest (first on ties) unit of another player, no active peace treaty with the
/// mover, Tentacles, not Frozen. Returns its tile and the full battle damage of it against the arrived unit.
pub fn tentacle_trap(s: &State, t: usize, mover: i32, r: &Rules) -> Option<(usize, f32)> {
    let u = &s.units;
    let size = s.size as i32;
    let (x, y) = ((t % s.size) as i32, (t / s.size) as i32);
    let mut best: Option<usize> = None;
    for (dx, dy) in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)] {
        let (nx, ny) = (x + dx, y + dy);
        if nx < 0 || ny < 0 || nx >= size || ny >= size {
            continue;
        }
        let j = (ny * size + nx) as usize;
        let o = u.owner[j];
        if o == 0 || o == mover || crate::diplomacy::protected(s, mover, o) {
            continue;
        }
        if !r.tentacles[u.kind[j].clamp(0, 127) as usize] || u.frozen[j] {
            continue;
        }
        if best.map_or(true, |b| u.health[b] < u.health[j]) {
            best = Some(j);
        }
    }
    let j = best?;
    let (kj, kt) = (u.kind[j] as usize, u.kind[t] as usize);
    let bonus = crate::military::defense_bonus(s, r, t);
    let (damage, _, _) = crate::kernels::combat(r.unit_attack[kj], u.health[j], crate::kernels::max_health(s, r, j),
                                                r.unit_defense[kt], u.health[t], crate::kernels::max_health(s, r, t), bonus);
    Some((j, damage))
}

/// The trap's attack action (Passive, no retaliation) -> perform attack: both units turn, the target loses Boosted and
/// Bubble and takes the damage; a kill credits the Tentacles unit's owner (no xp: Static).
pub fn trap_attack(s: &mut State, jelly: usize, t: usize, damage: f32, r: &Rules) {
    if s.units.owner[t] == 0 {
        return;
    }
    let size = s.size;
    let present = s.units.owner[jelly] != 0;
    if present {
        s.units.direction[jelly] = crate::military::direction(size, jelly, t);
    }
    s.units.direction[t] = crate::military::direction(size, t, jelly);
    s.units.boosted[t] = false;
    s.units.bubble[t] = false;
    s.units.swift[t] = false;
    s.units.health[t] -= damage;
    if s.units.health[t] <= 0.0 {
        crate::military::remove(s, t, r, if present { Some(jelly) } else { None }, true);
    }
}

/// get unit actions' Build of the Manual Canal (52): the mover's
/// unit with can build (not moved, not attacked, not Frozen), Waterways, and can build for Canal on any
/// owner's tile (Freelance, Manual), any improvement there (Discrete), not Flooded, Field or Forest: `floodable`.
pub fn can_canal(s: &State, i: usize) -> bool {
    let u = &s.units;
    u.owner[i] == s.current_player && u.owner[i] != 0 && !u.moved[i] && !u.attacked[i] && !u.frozen[i]
        && s.players.tech[s.p()][WATERWAYS] && floodable(s, i)
}

/// build command(Canal) -> build action: no improvement placed, cost 0, no population;
/// flood tile action; the unit on the tile is moved and attacked, an invisible one revealed (after the flood).
pub fn canal_command(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    let player = s.current_player;
    s.units.moved[i] = true;
    s.units.attacked[i] = true;
    s.units.move_lock[i] = false;
    flood_tile(s, i, player, r, g);
    s.units.invisible[i] = false;
}

/// get unit actions' clear tile effect Flooded: the mover's unit can attack and move (not
/// moved, not attacked, not Frozen), its tile is Flooded, the player lacks the Flooded tribe ability (is not
/// Aquarion), and the unit is Amphibious or has no Swim.
pub fn can_clear_flood(s: &State, i: usize, r: &Rules) -> bool {
    let u = &s.units;
    let kind = u.kind[i].clamp(0, 127) as usize;
    u.owner[i] == s.current_player && u.owner[i] != 0 && !u.moved[i] && !u.attacked[i] && !u.frozen[i]
        && s.tiles.flooded[i] && s.players.tribe[s.p()] != crate::tribes::AQUARION
        && (r.amphibious[kind] || !r.skills[kind][FLOAT])
}

/// clear tile effect command -> clear tile effect action(Flooded, by unit): the
/// effect goes; update routes then update city connections(player); the unit is exhausted (moved, attacked, Boosted
/// lost). No command base: nothing is deferred (and nothing is explored).
pub fn clear_flood_command(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    let player = s.current_player;
    s.tiles.flooded[i] = false;
    crate::structures::update_routes(s, r);
    crate::structures::connections(s, player, g, true);
    s.units.moved[i] = true;
    s.units.attacked[i] = true;
    s.units.move_lock[i] = false;
    s.units.boosted[i] = false;
    s.units.double_ready[i] = false;     // make exhausted
}
