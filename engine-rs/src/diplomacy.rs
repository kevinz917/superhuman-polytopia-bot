//! gpu/diplomacy.py: native two-player diplomacy. Games with more seats have no diplomacy (no ability is legal);
//! their per-pair tables stay false, so the peace terms elsewhere are inert.

use crate::economy::discover;
use crate::geom::Geom;
use crate::military::remove;
use crate::rules::Rules;
use crate::state::State;
use crate::structures::connections;

#[inline]
pub fn protected(s: &State, owner: i32, other: i32) -> bool {
    let (p, q) = (s.pclip(owner), s.pclip(other));
    other > 0 && (owner == other || s.players.peace[p][q] || s.players.truce_breaker[p][q] != 0)
}

/// Opponent capital tile (first match) and whether one exists; shared by legal_masks and decode.
pub fn opponent_capital(s: &State) -> (usize, bool) {
    if s.np != 2 {
        return (0, false);
    }
    let other = 3 - s.current_player;
    for i in 0..s.n {
        if s.tiles.capital[i] == other && s.cities.owner[i] == other {
            return (i, true);
        }
    }
    (0, false)
}

/// diplomacy.legal for Action(3, source, target, ability).
pub fn legal(s: &State, source: i32, target: i32, ability: i32) -> bool {
    legal_with(s, source, target, ability, opponent_capital(s))
}

/// diplomacy.legal given opponent_capital(s).
pub fn legal_with(s: &State, source: i32, target: i32, ability: i32, (capital, has_capital): (usize, bool)) -> bool {
    if s.np != 2 {
        return false;
    }
    let p = s.p();
    let q = (2 - s.current_player) as usize;
    let pl = &s.players;
    let other = q as i32 + 1;
    let alive = source == other && pl.killed_turn[q] == 0 && pl.resigned_turn[q] == 0;
    let eligible = alive && pl.known[p][q];
    // a pending peace request is a command trigger: it must be answered before anything else
    // (the turn start's diplomacy check makes it pending on the receiver's turn)
    if peace_pending(s) {
        return alive && ability == 19 && (target == 0 || target == 1);
    }
    eligible
        && ((ability == 18 && pl.peace[p][q])
            // establishing an embassy: no war check; the cost is embassy_cost
            || (ability == 20 && pl.tech[p][38] && !pl.embassy[p][q] && has_capital && target == capital as i32
                && pl.stars[p] >= embassy_cost(s, p))
            // a peace treaty: no cool-down check
            || (ability == 21 && pl.tech[p][7] && !pl.peace[p][q] && !pl.peace_offer[q][p] && !pl.peace_offer[p][q])
            || (ability == 22 && pl.embassy[p][q]))
}

/// The mover has a peace request to answer (two seats; `peace_offer[p][q]`: p was offered peace by q).
#[inline]
pub fn peace_pending(s: &State) -> bool {
    s.np == 2 && s.players.peace_offer[s.p()][(2 - s.current_player) as usize]
}

/// Embassy cost: 4 + 2 per embassy the player already has.
pub fn embassy_cost(s: &State, p: usize) -> i32 {
    4 + 2 * (0..s.np).filter(|&q| q != p && s.players.embassy[p][q]).count() as i32
}

/// Seat `p` is at war with seat `q`: not at peace, and either attacked the
/// other within the last round (from that attack up to, not including, the attacker's next turn).
pub fn at_war(s: &State, p: usize, q: usize) -> bool {
    let now = s.turn * s.np as i32 + s.p() as i32;
    !s.players.peace[p][q] && (now < s.players.hostile_until[p] || now < s.players.hostile_until[q])
}

/// An attack by the mover changes the two sides' relations: stamps the attack (war until the
/// mover's next turn) and destroys the embassies between the two sides. Pending peace requests stay.
pub fn hostile(s: &mut State) {
    let p = s.p();
    s.players.embassy = Default::default();
    s.players.hostile_until[p] = s.turn * s.np as i32 + p as i32 + s.np as i32;
}

pub fn capital_vision(s: &mut State) {
    let n = s.n;
    let p = s.p();
    if !s.players.tech[p][38] {
        return; // discover of an empty reveal changes nothing
    }
    // the capitals of every met seat (with two seats: the opponent's, once met)
    let met: Vec<i32> = (0..s.np).filter(|&q| q != p && s.players.known[p][q]).map(|q| q as i32 + 1).collect();
    if met.is_empty() {
        return;
    }
    let reveal: Vec<bool> = (0..n).map(|t| met.contains(&s.tiles.capital[t])).collect();
    discover(s, &reveal, s.current_player);
}

pub fn start_turn(s: &mut State, owner: i32) {
    for a in 0..s.np {
        for b in 0..s.np {
            if s.players.truce_breaker[a][b] == owner {
                s.players.truce_breaker[a][b] = 0;
            }
        }
    }
}

pub fn execute(s: &mut State, target: i32, ability: i32, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let p = s.p();
    let q = (2 - cp) as usize;
    match ability {
        18 => {
            s.players.peace[p][q] = false;
            s.players.peace[q][p] = false;
            s.players.truce_breaker[p][q] = cp;
            s.players.truce_breaker[q][p] = cp;
            let doomed: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] == cp && s.tiles.owner[i] == q as i32 + 1).collect();
            for i in doomed {
                remove(s, i, r, None, true);
            }
            for i in 0..n {
                if s.units.owner[i] == cp {
                    s.units.moved[i] = true;
                    s.units.attacked[i] = true;
                }
            }
        }
        19 => {
            let accepted = target == 1;
            s.players.peace[p][q] = accepted;
            s.players.peace[q][p] = accepted;
            s.players.peace_offer[p][q] = false;
            s.players.peace_offer[q][p] = false;
        }
        20 => {
            s.players.stars[p] -= embassy_cost(s, p);
            s.players.embassy[p][q] = true;
            // establishing an embassy: the host meets the builder (star gift included)
            crate::core::meet_pair(s, q, p);
            let t = target.clamp(0, n as i32 - 1) as usize;
            let reveal: Vec<bool> = (0..n).map(|i| g.d(t, i) <= 1).collect();
            discover(s, &reveal, cp);
        }
        21 => {
            s.players.peace_offer[q][p] = true;
        }
        _ => {
            s.players.embassy[p][q] = false;
        }
    }
    connections(s, cp, g, true);
    connections(s, 3 - cp, g, true);
}
