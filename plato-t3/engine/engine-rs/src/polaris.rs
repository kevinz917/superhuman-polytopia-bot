//! Polaris (increment S1): freezing. Rules as in the
//! project's full-game engine port.
//!
//! Representation: a tile is frozen (is frozen) when its terrain is Ice or it is polar land
//! (`tiles.polar`: climate 16). Skins are 0 in our games, so the skin tests reduce to the climate test.
use crate::diplomacy::{hostile, protected};
use crate::geom::Geom;
use crate::kernels::ICE;
use crate::rules::Rules;
use crate::state::State;

#[inline]
fn water(terrain: i32) -> bool {
    terrain == 1 || terrain == 2
}

/// is frozen.
#[inline]
pub fn frozen_tile(s: &State, t: usize) -> bool {
    s.tiles.terrain[t] == ICE || (s.tiles.polar[t] && !water(s.tiles.terrain[t]))
}

/// is freezable for `player`: a unit with no active peace treaty with the player (own units, peace and
/// cool-down count as one) and not already Frozen.
#[inline]
pub fn unit_freezable(s: &State, t: usize, player: i32) -> bool {
    let o = s.units.owner[t];
    o > 0 && !protected(s, player, o) && !s.units.frozen[t]
}

/// is freezable(state, player, only owned): explored by the player (and owned, with `only_owned`), and water,
/// or land not of the player's climate, or a visible freezable unit on it.
pub fn tile_freezable(s: &State, t: usize, player: i32, only_owned: bool) -> bool {
    let p = s.pclip(player);
    if (only_owned && s.tiles.owner[t] != player) || !s.tiles.explored[p][t] {
        return false;
    }
    // a Flooded tile stays freezable (freeze tile removes Swamped only)
    water(s.tiles.terrain[t]) || s.tiles.flooded[t] || !s.tiles.polar[t] || (!s.hidden_from(t, player) && unit_freezable(s, t, player))
}

/// freeze tile (v >= 120) as freeze tile action runs it: re-checks is freezable (only owned false); water
/// becomes Ice; the tile takes the player's climate (polar: the freezing player is Polaris, or a converted
/// freezer, which this engine also marks polar). Returns whether the tile changed.
pub fn freeze_tile(s: &mut State, t: usize, player: i32) -> bool {
    if !tile_freezable(s, t, player, false) {
        return false;
    }
    if water(s.tiles.terrain[t]) {
        s.tiles.terrain[t] = ICE;
    }
    s.tiles.polar[t] = true;
    true
}

/// freeze unit action with damage 0 and no origin (FreezeArea, AutoFreeze): modify diplomacy relations from attack, the
/// unit is moved, attacked and Frozen, and the freezing player's Pacifist task is reset.
pub fn freeze_unit(s: &mut State, t: usize, player: i32) {
    let victim = s.units.owner[t];
    if victim <= 0 {
        return;
    }
    hostile(s, victim);
    s.units.moved[t] = true;
    s.units.attacked[t] = true;
    s.units.move_lock[t] = false;
    s.units.frozen[t] = true;
    let p = s.pclip(player);
    s.players.attacked[p] = true;
}

/// freeze area action: over the radius area, centre included, each freezable raw unit is
/// frozen, then each freezable tile; then update routes and update city connections of the player run last. The
/// per-tile effects commute, so the sorted order does not matter here.
pub fn freeze_area(s: &mut State, centre: usize, radius: i32, freeze_units: bool, only_owned: bool, r: &Rules, g: &Geom) {
    let player = s.current_player;
    let n = s.n;
    let area: Vec<usize> = (0..n).filter(|&t| g.d(centre, t) <= radius).collect();
    if area.is_empty() {
        return;
    }
    // pushed before the freeze tile per tile (LIFO): the push-time test decides which tiles get a freeze tile
    let pushed: Vec<bool> = area.iter().map(|&t| tile_freezable(s, t, player, only_owned)).collect();
    for (k, &t) in area.iter().enumerate() {
        if freeze_units && unit_freezable(s, t, player) {
            freeze_unit(s, t, player);
        }
        if pushed[k] {
            freeze_tile(s, t, player);
        }
    }
    sync_ice_banks(s);
    crate::structures::update_routes(s, r);
    crate::structures::connections(s, player, g, true);
}

/// move action (v > 101, not a push): an AutoFreeze unit (Mooni, Gaami) freezes radius 1 around where
/// it arrives, units included.
pub fn auto_freeze(s: &mut State, at: usize, kind: usize, r: &Rules, g: &Geom) {
    if r.autofreeze[kind] {
        freeze_area(s, at, 1, true, false, r, g);
    }
}

/// restore climate (v > 111) on a tile of the Polaris climate, for a player whose climate does not spread
/// (any non-Polaris player): polar land other than Ice goes back to the player's climate. Ice keeps it. For Polaris
/// it compares skins only, which are all 0 here: nothing.
#[inline]
pub fn restore_climate(s: &mut State, t: usize, player: i32) {
    if s.players.tribe[s.pclip(player)] != crate::tribes::POLARIS && s.tiles.polar[t]
        && !water(s.tiles.terrain[t]) && s.tiles.terrain[t] != ICE {
        s.tiles.polar[t] = false;
    }
}

/// The city area's climate after an expand city / capture city of `player`: Polaris (Spread, tribe 16) freezes every
/// tile ruled by the city (freeze tile), any other player restores the climate there.
pub fn city_area_climate(s: &mut State, city: usize, player: i32) {
    let polaris = s.players.tribe[s.pclip(player)] == crate::tribes::POLARIS;
    for t in 0..s.n {
        if s.tiles.city[t] == city as i32 {
            if polaris {
                freeze_tile(s, t, player);
            } else {
                restore_climate(s, t, player);
            }
        }
    }
    if polaris {
        sync_ice_banks(s);
    }
}

/// get ice bank level: min(count ice tiles / 20, max level 30), count ice tiles over the whole map.
pub fn ice_bank_level(s: &State) -> i32 {
    ((0..s.n).filter(|&t| frozen_tile(s, t)).count() as i32 / 20).min(30)
}

/// check ice bank levels (and the owner's start turn update improvement level): every Ice Bank's stored level becomes the
/// calculated one (its level actions carry no growth reward, so nothing else changes).
pub fn sync_ice_banks(s: &mut State) {
    if !s.tiles.structure.contains(&34) {
        return;
    }
    let level = ice_bank_level(s);
    for t in 0..s.n {
        if s.tiles.structure[t] == 34 {
            s.tiles.structure_level[t] = level;
        }
    }
}

/// start turn action's Ice Bank income for `owner`: calculate work = work 2 x the calculated level, per owned bank.
pub fn ice_bank_income(s: &State, owner: i32) -> i32 {
    let banks = (0..s.n).filter(|&t| s.tiles.structure[t] == 34 && s.tiles.owner[t] == owner).count() as i32;
    if banks == 0 { 0 } else { banks * 2 * ice_bank_level(s) }
}

/// can freeze area(unit, 1): not moved, not attacked, the FreezeArea ability, and a freezable tile
/// (is freezable, only owned false) in the radius-1 area around it, centre excluded.
pub fn can_freeze_area(s: &State, i: usize, r: &Rules, g: &Geom) -> bool {
    let u = &s.units;
    let kind = u.kind[i] as usize;
    let owner = u.owner[i];
    if owner != s.current_player || u.moved[i] || u.attacked[i] || !r.freeze_area[kind] {
        return false;
    }
    let mut any = false;
    g.for_ball(i, 1, |t| any |= t != i && tile_freezable(s, t, owner, false));
    any
}

/// freeze area command: the unit is moved and attacked, then freeze area action(radius 1, units, any tile).
pub fn freeze_area_command(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    s.units.moved[i] = true;
    s.units.attacked[i] = true;
    s.units.move_lock[i] = false;
    freeze_area(s, i, 1, true, false, r, g);
}

/// has breakable ice(tile, player) (v >= 116): Ice, and a Bridge on it, or no unit, or a Swim / Fly unit, or a
/// unit the player cannot see (Fly: Elyrion's dragons).
pub fn has_breakable_ice(s: &State, t: usize, player: i32, r: &Rules) -> bool {
    if s.tiles.terrain[t] != ICE {
        return false;
    }
    let o = s.units.owner[t];
    let k = s.units.kind[t] as usize;
    s.tiles.structure[t] == 48 || s.tiles.algae[t] || o == 0 || r.skills[k][crate::rules::FLOAT] || r.fly[k] || s.hidden_from(t, player)
}

/// can break ice (v >= 99): any tribe's unit, not attacked, not moved unless Dash, no AutoFreeze, and
/// breakable ice in the radius-1 area around it, centre excluded.
pub fn can_break_ice(s: &State, i: usize, r: &Rules, g: &Geom) -> bool {
    let u = &s.units;
    let kind = u.kind[i] as usize;
    let owner = u.owner[i];
    // can break ice: no leader (Cymanti)
    if owner != s.current_player || u.attacked[i] || (u.moved[i] && !r.dash[kind]) || r.autofreeze[kind] || u.leader[i] != 0 {
        return false;
    }
    let mut any = false;
    g.for_ball(i, 1, |t| any |= t != i && has_breakable_ice(s, t, owner, r));
    any
}

/// break ice command: the unit is moved and attacked; an invisible breaker is revealed (after the breaking);
/// break ice area action over radius 1, centre included, nearest first: each break ice action on breakable ice reveals an
/// unseen unit there and stops, or turns the Ice into Water (an orthogonal neighbour is land other than Ice) or
/// Ocean; then check ice bank levels; update routes and update city connections of the player run last.
pub fn break_ice_command(s: &mut State, i: usize, r: &Rules, g: &Geom) {
    let player = s.current_player;
    s.units.moved[i] = true;
    s.units.attacked[i] = true;
    s.units.move_lock[i] = false;
    let reveal_self = s.units.invisible[i];
    let size = s.size;
    let mut area: Vec<usize> = Vec::new();
    g.for_ball(i, 1, |t| area.push(t));
    let pushed: Vec<usize> = area.into_iter().filter(|&t| has_breakable_ice(s, t, player, r)).collect();
    for t in pushed {
        if s.units.owner[t] != 0 && s.hidden_from(t, player) {
            s.units.invisible[t] = false;
            continue;
        }
        if !has_breakable_ice(s, t, player, r) {
            continue;
        }
        let (x, y) = ((t % size) as i32, (t / size) as i32);
        let land_next = [(0, -1), (-1, 0), (1, 0), (0, 1)].iter().any(|&(dx, dy)| {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= size as i32 || ny >= size as i32 {
                return false;
            }
            let k = s.tiles.terrain[(ny as usize) * size + nx as usize];
            // map data is_land (3..=6) other than Ice
            (3..=5).contains(&k)
        });
        s.tiles.terrain[t] = if land_next { 1 } else { 2 };
    }
    sync_ice_banks(s);
    if reveal_self {
        s.units.invisible[i] = false;
    }
    crate::structures::update_routes(s, r);
    crate::structures::connections(s, player, g, true);
}
