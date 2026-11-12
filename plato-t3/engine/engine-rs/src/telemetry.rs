//! Diagnostic counters, never policy inputs or physical State exports. Native engine turns are zero-based.
use crate::core::Action;
use crate::state::{State, MAXP};
pub const WIDTH: usize = 27;
pub const FIELDS: [&str; WIDTH] = ["diplomacy_research_turn", "riding_research_turn", "chivalry_research_turn", "cloaks_trained", "mind_benders_trained", "riders_trained", "knights_trained", "embassies", "peace_offered", "peace_accepted", "peace_broken", "city_reward_1", "city_reward_2", "city_reward_3", "city_reward_4", "city_reward_5", "city_reward_6", "city_reward_7", "city_reward_8", "reward_choice", "reward_explored_tiles", "reward_city", "reward_turn", "complete", "diplomacy_at_start", "riding_at_start", "chivalry_at_start"];
#[derive(Clone, Copy)]
pub struct Metrics { pub values: [[i32; WIDTH]; MAXP] }
impl Metrics {
    pub fn new() -> Self {
        let mut values = [[0; WIDTH]; MAXP];
        for row in &mut values { row[..3].fill(-1); }
        Self { values }
    }
}
pub fn initialize(s: &mut State, complete: bool) {
    s.metrics = Metrics::new();
    for p in 0..s.np {
        for (k, &tech) in [38, 1, 3].iter().enumerate() {
            s.metrics.values[p][24 + k] = if complete { i32::from(s.players.tech[p][tech]) } else { -1 };
            if s.players.tech[p][tech] { s.metrics.values[p][k] = s.players.tech_turn[p][tech]; }
        }
        s.metrics.values[p][23] = i32::from(complete);
    }
}
pub fn clear_event(s: &mut State) { for row in &mut s.metrics.values { row[19..23].fill(0); } }
pub fn record(s: &mut State, a: Action) {
    let p = s.p();
    let explored = if a.kind == 9 { (0..s.n).filter(|&i| s.tiles.explored[p][i] || !s.fog).count() as i32 } else { 0 };
    let m = &mut s.metrics.values[p];
    match a.kind {
        7 => for (k, &tech) in [38, 1, 3].iter().enumerate() { if a.parameter == tech && m[k] < 0 { m[k] = s.turn; } },
        4 => { match a.parameter { 38 | 39 | 50 => m[3] += 1, 10 => m[4] += 1, 3 => m[5] += 1, 4 => m[6] += 1, _ => {} } },
        3 => { match a.parameter { 20 => m[7] += 1, 21 => m[8] += 1, 19 if a.target == 1 => m[9] += 1, 18 => m[10] += 1, _ => {} } },
        9 if (1..=8).contains(&a.parameter) => {
            m[10 + a.parameter as usize] += 1;
            m[19] = a.parameter; m[20] = explored; m[21] = a.target; m[22] = s.turn;
        },
        _ => {}
    }
}
