//! Optional wall-clock profile of tick and step phases (diagnostic, off unless GameBatch.profile_phases(True)).

use std::cell::{Cell, RefCell};
use std::time::Instant;

pub const KINDS: [&str; 18] = [
    "research", "recover", "move", "end_turn", "attack", "summon", "harvest", "forest", "promote", "disband", "reward",
    "build", "destroy", "capture", "heal", "diplomacy", "resign", "unsupported",
];
/// Phase ids: 0..18 step validation per kind, 18..36 step execution per kind, then the fixed phases below.
pub const VALIDATE: usize = 0;
pub const EXECUTE: usize = 18;
pub const STEP_PRE: usize = 36;
pub const STEP_POST: usize = 37;
pub const TICK_STEP: usize = 38;
pub const TICK_RESET: usize = 39;
pub const TICK_MASKS: usize = 40;
pub const TICK_OBSERVE: usize = 41;
pub const TICK_OBSERVE_OPP: usize = 42;
pub const TICK_FINAL: usize = 43;
pub const TICK_ALLOC: usize = 44;
pub const TICK_OUTPUT: usize = 45;
pub const N: usize = 46;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static TOTAL: RefCell<[(u64, u64); N]> = const { RefCell::new([(0, 0); N]) };
}

pub fn enable(on: bool) {
    ON.with(|c| c.set(on));
    TOTAL.with(|t| *t.borrow_mut() = [(0, 0); N]);
}

/// Run `f`, adding its wall time to phase `id` when profiling is on.
#[inline]
pub fn time<T>(id: usize, f: impl FnOnce() -> T) -> T {
    if !ON.with(|c| c.get()) {
        return f();
    }
    let t0 = Instant::now();
    let out = f();
    let ns = t0.elapsed().as_nanos() as u64;
    TOTAL.with(|t| {
        let mut t = t.borrow_mut();
        t[id].0 += 1;
        t[id].1 += ns;
    });
    out
}

pub fn name(id: usize) -> String {
    match id {
        i if i < EXECUTE => format!("validate.{}", KINDS[i]),
        i if i < STEP_PRE => format!("execute.{}", KINDS[i - EXECUTE]),
        STEP_PRE => "step.pre".into(),
        STEP_POST => "step.post".into(),
        TICK_STEP => "tick.step".into(),
        TICK_RESET => "tick.reset".into(),
        TICK_MASKS => "tick.masks".into(),
        TICK_OBSERVE => "tick.observe".into(),
        TICK_OBSERVE_OPP => "tick.observe_opp".into(),
        TICK_FINAL => "tick.final_view".into(),
        TICK_ALLOC => "tick.alloc".into(),
        _ => "tick.output".into(),
    }
}

/// (name, calls, total ms) for every phase that ran.
pub fn snapshot() -> Vec<(String, u64, f64)> {
    TOTAL.with(|t| {
        t.borrow().iter().enumerate().filter(|(_, v)| v.0 > 0).map(|(i, v)| (name(i), v.0, v.1 as f64 / 1e6)).collect()
    })
}
