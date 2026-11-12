//! Diplomacy between every pair of seats, two seats or four (the real game's player diplomacy extensions,
//! get diplomacy commands for opponent and the diplomacy commands; docs/rules-parity-2-actions.md).
//!
//! The diplomatic action family has one block of BLOCK entries per other seat, addressed by turn-order offset from
//! the mover: block b is the seat b + 1 places after the mover (block_seat). A two-seat game only ever uses block 0,
//! the sole opponent, so its entries keep the earlier one-opponent layout. Within a block: 0 break peace,
//! 1 decline / 2 accept a peace request, 3 embassy, 4 offer peace, 5 destroy embassy.

use crate::economy::discover;
use crate::geom::Geom;
use crate::military::remove;
use crate::rules::Rules;
use crate::state::{State, MAXP};
use crate::structures::{connections, update_routes};

/// Entries per opponent block of the diplomatic family.
pub const BLOCK: usize = 6;
/// Blocks in the family (one per possible other seat).
pub const BLOCKS: usize = MAXP - 1;
/// Action parameter (ability) of each entry of a block.
pub const ABILITIES: [i32; BLOCK] = [18, 19, 19, 20, 21, 22];

#[inline]
pub fn protected(s: &State, owner: i32, other: i32) -> bool {
    let (p, q) = (s.pclip(owner), s.pclip(other));
    other > 0 && (owner == other || s.players.peace[p][q] || s.players.truce_breaker[p][q] != 0)
}

/// Seat id of block `b`: the seat b + 1 places after the mover in turn order; 0 when the game has no such seat.
#[inline]
pub fn block_seat(s: &State, b: usize) -> i32 {
    if b + 1 >= s.np {
        return 0;
    }
    (s.current_player - 1 + b as i32 + 1).rem_euclid(s.np as i32) + 1
}

/// Block of seat `q` (1-based id, not the mover) relative to the mover.
#[inline]
pub fn seat_block(s: &State, q: i32) -> usize {
    ((q - s.current_player).rem_euclid(s.np as i32) - 1) as usize
}

/// Seat `q`'s capital tile (its original capital, still its own city; first match) and whether it has one
/// (get current capital coordinates / owns their capital).
pub fn capital_of(s: &State, q: i32) -> (usize, bool) {
    if q <= 0 {
        return (0, false);
    }
    for i in 0..s.n {
        if s.tiles.capital[i] == q && s.cities.owner[i] == q {
            return (i, true);
        }
    }
    (0, false)
}

/// The two-seat opponent's capital (block 0's embassy target).
pub fn opponent_capital(s: &State) -> (usize, bool) {
    capital_of(s, block_seat(s, 0))
}

/// diplomacy.legal for Action(3, source = the other seat's id, target, ability).
pub fn legal(s: &State, source: i32, target: i32, ability: i32) -> bool {
    legal_with(s, source, target, ability, capital_of(s, source))
}

/// diplomacy.legal given capital_of(s, source).
pub fn legal_with(s: &State, source: i32, target: i32, ability: i32, (capital, has_capital): (usize, bool)) -> bool {
    let np = s.np as i32;
    if source < 1 || source > np || source == s.current_player {
        return false;
    }
    // a pending peace request is a command trigger: it must be answered before anything else
    // (peace request response command; check diplomacy makes it pending on the receiver's turn)
    if let Some(sender) = pending_sender(s) {
        return source == sender && ability == 19 && (target == 0 || target == 1);
    }
    let p = s.p();
    let q = (source - 1) as usize;
    let pl = &s.players;
    // get diplomacy commands for opponent: only known players still in the game
    let eligible = s.alive(q) && pl.known[p][q];
    eligible
        // break peace command validity
        && ((ability == 18 && pl.peace[p][q])
            // establish embassy command validity (v >= 80): no war check; get embassy cost for player
            || (ability == 20 && pl.tech[p][38] && !pl.embassy[p][q] && has_capital && target == capital as i32
                && pl.stars[p] >= embassy_cost(s, p))
            // peace treaty command validity: no cool-down check; no request of ours already waiting at theirs
            || (ability == 21 && pl.tech[p][7] && !pl.peace[p][q] && !pl.peace_offer[q][p] && !pl.peace_offer[p][q]))
    // ability 22 (destroy embassy command) is never offered: get diplomacy commands for opponent lists only peace treaty /
    // break peace and establish embassy; embassies go only through an attack
    // (modify diplomacy relations from attack) or a wipe. peace-on lockstep: engine-rs offered it.
}

/// The seat whose peace request the mover must answer now, if any (`peace_offer[p][q]`: p was offered peace by q).
/// check diplomacy pushes one receive diplomacy message action per message onto the LIFO action stack,
/// from the last message to the first, and the stack stops while a trigger is pending: the requests are answered
/// one at a time, oldest first. All of them were sent since the mover's previous turn, by seats moving in turn
/// order after it, so the oldest is the nearest such seat in turn order. Requests of seats no longer in the game
/// are dropped (v >= 82).
pub fn pending_sender(s: &State) -> Option<i32> {
    let p = s.p();
    (0..s.np - 1).map(|b| block_seat(s, b)).find(|&q| {
        let qi = (q - 1) as usize;
        s.players.peace_offer[p][qi] && s.alive(qi)
    })
}

/// The mover has a peace request to answer.
#[inline]
pub fn peace_pending(s: &State) -> bool {
    pending_sender(s).is_some()
}

/// get embassy cost for player (v > 113): 4 + 2 per embassy the player already has.
pub fn embassy_cost(s: &State, p: usize) -> i32 {
    4 + 2 * (0..s.np).filter(|&q| q != p && s.players.embassy[p][q]).count() as i32
}

/// Seat `p` is at war with seat `q` (has war with): not at peace, and either attacked the
/// other within the last round: from that attack through the attacker's next turn, its start included
/// (attacked_within_turn: turn L, or turn L + 1 while the current index is not past the attacker's).
/// `hostile_until` is the attacker's next-turn serial; 0 = never attacked.
pub fn at_war(s: &State, p: usize, q: usize) -> bool {
    let now = s.turn * s.np as i32 + s.p() as i32;
    let h = &s.players.hostile_until;
    let war = |x: i32| x > 0 && now <= x;
    !s.players.peace[p][q] && (war(h[p][q]) || war(h[q][p]))
}

/// modify diplomacy relations from attack(mover, victim): stamps the mover's attack on
/// `victim` (war until the mover's next turn) and destroys the embassies between the two (destroy embassy action both
/// ways). Other pairs and pending peace requests are untouched. No-op for neutral or the mover itself.
pub fn hostile(s: &mut State, victim: i32) {
    if victim <= 0 || victim > s.np as i32 || victim == s.current_player {
        return;
    }
    let p = s.p();
    let q = (victim - 1) as usize;
    s.players.embassy[p][q] = false;
    s.players.embassy[q][p] = false;
    s.players.hostile_until[p][q] = s.turn * s.np as i32 + p as i32 + s.np as i32;
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

/// check diplomacy for seat `owner` (after the turn counter moved on): requests from seats no longer
/// in the game are dropped, and each of the seat's cool down relations ends on its own clock: towards a lower-id seat
/// once the break is in an earlier turn, towards a higher-id seat once it is two turns back (State -> Neutral).
pub fn start_turn(s: &mut State, owner: i32) {
    let p = (owner - 1) as usize;
    for q in 0..s.np {
        if q == p {
            continue;
        }
        if s.players.peace_offer[p][q] && !s.alive(q) {
            s.players.peace_offer[p][q] = false;
        }
        if s.players.truce_breaker[p][q] != 0 {
            let broken = s.players.peace_broken[p][q];
            if (q < p && broken < s.turn) || (q > p && broken + 1 < s.turn) {
                s.players.truce_breaker[p][q] = 0;
            }
        }
    }
}

/// Execute a legal diplomatic action of the mover towards seat `source` (1-based id).
pub fn execute(s: &mut State, source: i32, target: i32, ability: i32, r: &Rules, g: &Geom) {
    let n = s.n;
    let cp = s.current_player;
    let p = s.p();
    let q = (source - 1) as usize;
    match ability {
        18 => {
            // break peace action (v >= 80): both relations cool down, both last peace broken turn; the breaker's units in
            // the victim's territory are disbanded, every other unit of the breaker is exhausted
            s.players.peace[p][q] = false;
            s.players.peace[q][p] = false;
            s.players.truce_breaker[p][q] = cp;
            s.players.truce_breaker[q][p] = cp;
            s.players.peace_broken[p][q] = s.turn;
            s.players.peace_broken[q][p] = s.turn;
            let doomed: Vec<usize> = (0..n).filter(|&i| s.units.owner[i] == cp && s.tiles.owner[i] == source).collect();
            for i in doomed {
                if s.units.owner[i] != cp {
                    continue;   // a follower already disbanded with its chain
                }
                // a disband: no spores; (Cymanti) a follower gets its own (paid) disband unit action
                s.units.poison[i] = false;
                let follower = crate::cymanti::unlink(s, i);
                remove(s, i, r, None, true);
                if let Some(f) = follower {
                    if s.units.owner[f] != 0 {
                        let p = s.p();
                        crate::military::disband_chain(s, f, r, p, true);
                    }
                }
            }
            for i in 0..n {
                if s.units.owner[i] == cp {
                    s.units.moved[i] = true;
                    s.units.attacked[i] = true;
                    s.units.move_lock[i] = false;
                }
            }
        }
        19 => {
            // peace request response command pops the pending trigger; peace request response action sets both relations to
            // Peace when accepted (which ends a cool down)
            if target == 1 {
                s.players.peace[p][q] = true;
                s.players.peace[q][p] = true;
                s.players.truce_breaker[p][q] = 0;
                s.players.truce_breaker[q][p] = 0;
            }
            s.players.peace_offer[p][q] = false;
        }
        20 => {
            s.players.stars[p] -= embassy_cost(s, p);
            s.players.embassy[p][q] = true;
            // establish embassy command: the host meets the builder (meet action, star gift included);
            // meet action's try reveal capital explores the builder's capital for a host with Capital Vision (tech 38)
            let new_meet = !s.players.known[q][p];
            crate::core::meet_pair(s, q, p);
            if new_meet && s.players.tech[q][38] {
                let reveal: Vec<bool> = (0..n).map(|i| s.tiles.capital[i] == cp).collect();
                discover(s, &reveal, source);
            }
            let t = target.clamp(0, n as i32 - 1) as usize;
            let reveal: Vec<bool> = (0..n).map(|i| g.d(t, i) <= 1).collect();
            discover(s, &reveal, cp);
        }
        21 => {
            // peace treaty action: a peace request message to the other seat, received at its next turn start
            s.players.peace_offer[q][p] = true;
        }
        _ => {
            s.players.embassy[p][q] = false;
        }
    }
    // break peace command and peace request response command (accepted only) queue update city connections action for
    // [the mover, the other player] under update routes action for each (
    // last in, first out: routes first). The other diplomatic commands update
    // no connections (an embassy's exploration does, through the deferred update in core::step_with). `source` is the
    // other seat of this pair (any seat count).
    if ability == 18 || (ability == 19 && target == 1) {
        update_routes(s, r);
        connections(s, cp, g, true);
        connections(s, source, g, true);
    }
}
