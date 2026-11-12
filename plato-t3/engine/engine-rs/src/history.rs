//! Observation history (observation version 3, 2026-11-01): what a seat remembers of its own past views.
//!
//! Everything here is a function of the seat's own observations, never of hidden state, so the policy gets no
//! information a human player could not have: a snapshot of what the seat sees is taken at the end of each of its
//! turns (`end_of_turn`), and at the start of its next turn (`start_of_turn`) the differences to that snapshot
//! are the "since my last turn" planes. Views then carry (kernels::observe):
//!   hist_enemy_unit[k]   a visible enemy unit stood here at the end of my turn k + 1 of my turns ago (k = 0, 1, 2)
//!   hist_enemy_built     an enemy structure, road or level appeared or changed here since my last turn (tile explored then)
//!   hist_enemy_trained   a visible enemy unit stands on an enemy city tile where none stood at the end of my last turn
//!   enemy_tech           techs the enemy must have: the tech of every enemy unit kind, structure and road seen, a unit
//!                        seen on a mountain (Climbing) or on water (Fishing), with every prerequisite (Rules::tech_closure)
//! In a game with more than two seats "enemy" is any other seat (one union over them).

use crate::rules::Rules;
use crate::state::{State, HIST_TURNS};

/// Tile code compared across turns: what the view shows of a tile's improvement.
#[inline]
fn code(s: &State, t: usize) -> i32 {
    s.tiles.structure[t] * 64 + s.tiles.structure_level[t].clamp(0, 31) * 2 + s.tiles.road[t] as i32
}

/// A visible, non-hidden enemy unit stands on t for `player`.
#[inline]
fn enemy_unit_visible(s: &State, player: i32, t: usize, vis: bool) -> bool {
    vis && State::foreign(s.units.owner[t], player) && !s.hidden_from(t, player)
}

/// Fold what `player` can see right now into its inferred enemy techs.
pub fn fold_tech(s: &mut State, player: i32, r: &Rules) {
    let p = (player - 1) as usize;
    let mut bits = s.history[p].enemy_tech;
    let mut raw = 0u128;
    for t in 0..s.n {
        let vis = s.tiles.explored[p][t] || !s.fog;
        if !vis {
            continue;
        }
        if enemy_unit_visible(s, player, t, true) {
            let kind = s.units.kind[t].clamp(0, 127) as usize;
            let tech = r.unit_tech[kind];
            if tech < 127 {
                raw |= 1u128 << tech;
            }
            if kind == 37 {
                raw |= 1u128 << 38; // a Dagger comes from a Cloak: Diplomacy
            }
            match s.tiles.terrain[t] {
                4 => raw |= 1u128 << 20, // a unit on a mountain: Climbing
                1 => raw |= 1u128 << 10, // a unit on shallow water: Fishing
                _ => {}
            }
        }
        if State::foreign(s.tiles.owner[t], player) {
            let st = s.tiles.structure[t].clamp(0, 127) as usize;
            let by = r.structures.unlock_by[st];
            if by != 0 && by & (by - 1) == 0 {
                raw |= by; // exactly one tech unlocks this structure
            }
            if s.tiles.road[t] {
                let by = r.structures.unlock_by[3];
                if by != 0 && by & (by - 1) == 0 {
                    raw |= by;
                }
            }
        }
    }
    while raw != 0 {
        let k = raw.trailing_zeros() as usize;
        raw &= raw - 1;
        bits |= r.tech_closure[k];
    }
    s.history[p].enemy_tech = bits & (r.tech_exists_bits | r.polaris.tech_exists_bits);
}

/// `player` ends its turn: remember what it sees now (shift the unit ring; record the tile codes).
pub fn end_of_turn(s: &mut State, player: i32, r: &Rules) {
    fold_tech(s, player, r);
    let p = (player - 1) as usize;
    let n = s.n;
    s.history[p].seen_unit.rotate_right(1); // [0] becomes the oldest slot, overwritten below
    for t in 0..n {
        let vis = s.tiles.explored[p][t] || !s.fog;
        let seen = enemy_unit_visible(s, player, t, vis);
        let c = if vis { code(s, t) } else { -1 };
        let h = &mut s.history[p];
        h.seen_unit[0][t] = seen;
        h.prev_code[t] = c;
    }
    debug_assert_eq!(s.history[p].seen_unit.len(), HIST_TURNS);
}

/// `player` starts its turn: what changed since the end of its last turn, on tiles it could see then.
pub fn start_of_turn(s: &mut State, player: i32, r: &Rules) {
    let p = (player - 1) as usize;
    let n = s.n;
    for t in 0..n {
        let vis = s.tiles.explored[p][t] || !s.fog;
        let prev = s.history[p].prev_code[t];
        let enemy_tile = State::foreign(s.tiles.owner[t], player);
        let built = vis && prev >= 0 && enemy_tile && code(s, t) != prev;
        let unit_now = enemy_unit_visible(s, player, t, vis);
        let trained = unit_now && !s.history[p].seen_unit[0][t] && State::foreign(s.cities.owner[t], player);
        let h = &mut s.history[p];
        h.built[t] = built;
        h.trained[t] = trained;
    }
    fold_tech(s, player, r);
}
