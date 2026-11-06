//! Branch-coverage counters (thread-local, diagnostic only): which rule paths a run exercised.

use std::cell::RefCell;

pub const NAMES: [&str; 40] = [
    "research", "recover", "move", "end_turn", "attack", "convert", "infiltrate", "summon", "upgrade", "harvest",
    "forest", "promote", "disband", "reward", "build", "destroy", "capture_city", "capture_enemy_city", "ruin_stars",
    "ruin_tech", "ruin_population", "ruin_explore", "ruin_unit", "starfish", "heal_others", "diplomacy", "resign",
    "splash", "stomp", "push", "elimination", "truncated_round_cap", "truncated_max_actions", "move_hidden_encounter",
    "retaliation_kill", "advance_after_kill", "embark_disembark", "area_damage_kill", "reward_giant", "reward_explore",
];

thread_local! {
    static COV: RefCell<[u64; 40]> = const { RefCell::new([0; 40]) };
}

#[inline]
pub fn hit(i: usize) {
    COV.with(|c| c.borrow_mut()[i] += 1);
}

pub fn snapshot() -> [u64; 40] {
    COV.with(|c| *c.borrow())
}

pub fn reset() {
    COV.with(|c| *c.borrow_mut() = [0; 40]);
}
