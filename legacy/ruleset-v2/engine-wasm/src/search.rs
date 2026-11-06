//! Leak-free turn search (rl/search.py TurnSearcher with depth=1, determinize=True, replies=0): every plan is
//! made on a world sampled from the searcher's own view (rl/belief.py::determinize). Lane 0 plays the greedy
//! policy's turn and lanes 1.. sample theirs; after END the opponent's turn is rolled out with the same network,
//! and a lane is scored by the critic at the searcher's next decision. The best lane's turn is committed on the
//! true game up to the first sub-action whose outcome the searcher sees differently (build_guarded_replay).
//!
//! The pieces are exported separately so the browser can spread lanes over Web Workers: a world is a pure
//! function of (state, seed), so every worker samples the same one.

use polytopia_engine_rs::core::{terminal_rewards, OK};
use polytopia_engine_rs::kernels::update_exploration;
use polytopia_engine_rs::mapgen;
use polytopia_engine_rs::state::State;
use polytopia_engine_rs::structures::income;
use serde_json::{json, Value};

use crate::policy::{Draw, Policy};
use crate::view::packed;
use crate::Engine;

const CROP: i32 = 2;
const METAL: i32 = 5;
/// rl/search.py defaults: a turn is at most 96 sub-actions, a depth-1 lane at most 96 * 3 engine steps.
pub const MAX_TURN_ACTIONS: usize = 96;
const MAX_STEPS: usize = MAX_TURN_ACTIONS * 3;
/// TurnSearcher.play_turn: after this many rejected samples in a row, commit one sub-action planned on the truth.
pub const MAX_REJECTED: usize = 8;

/// splitmix64: small, seedable, and the same in every browser.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A seed for one stream (a lane, a world) of a plan.
pub fn mix(seed: u64, a: u64, b: u64) -> u64 {
    Rng::new(seed ^ a.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ b.wrapping_mul(0xA076_1D64_78BD_642F)).next_u64()
}

/// Stream ids: the world of plan round r uses (r, WORLD), its lane l uses (r, l).
pub const WORLD: u64 = u64::MAX;

fn copy_city(dst: &mut State, j: usize, src: &State, i: usize) {
    let n = src.n;
    let (d, s) = (&mut dst.cities, &src.cities);
    d.owner[j] = s.owner[i];
    d.level[j] = s.level[i];
    d.population[j] = s.population[i];
    d.progress[j] = s.progress[i];
    d.production[j] = s.production[i];
    d.border[j] = s.border[i];
    d.rewards[j * 9..j * 9 + 9].copy_from_slice(&s.rewards[i * 9..i * 9 + 9]);
    d.territory[j * n..(j + 1) * n].copy_from_slice(&s.territory[i * n..(i + 1) * n]);
    d.connected[j] = s.connected[i];
    d.order[j] = s.order[i];
    d.infiltrated[j] = s.infiltrated[i];
    d.riot[j] = s.riot[i];
}

/// rl/belief.py::determinize: one world consistent with the mover's observation (see that module's docstring).
pub fn determinize(state: &State, e: &Engine, rng: &mut Rng) -> State {
    let (g, n) = (&e.geom, state.n);
    let d = |i: usize, j: usize| g.d(i, j);
    let me = state.current_player;
    let (p, q, enemy) = ((me - 1) as usize, (2 - me) as usize, 3 - me);
    let empty = State::empty(state.size);
    let (t, u, c, pl) = (&state.tiles, &state.units, &state.cities, &state.players);
    let seen: Vec<bool> = (0..n).map(|i| t.explored[p][i] || !state.fog).collect();
    let fog: Vec<bool> = seen.iter().map(|&x| !x).collect();
    let board = mapgen::layout((rng.next_u64() % i32::MAX as u64) as i32, g);

    // Cities: keep the searcher's and every enemy city with some seen territory; relocate the rest.
    let moved: Vec<bool> = (0..n).map(|i| {
        c.owner[i] == enemy && fog[i] && !(0..n).any(|j| (t.city[j] == i as i32 || c.territory[i * n + j]) && seen[j])
    }).collect();
    let kept_city: Vec<bool> = (0..n).map(|i| c.owner[i] > 0 && !moved[i]).collect();
    let kept_rule: Vec<bool> = (0..n).map(|j| {
        t.city[j] >= 0 && kept_city[t.city[j].clamp(0, n as i32 - 1) as usize] && t.owner[j] > 0
    }).collect();

    // Unseen tiles: the sampled layout, unowned, except the territory of kept cities.
    let mut s = state.clone();
    let st = &mut s.tiles;
    for j in (0..n).filter(|&j| fog[j]) {
        let village_or_ruin = board.structure[j] == 1 || board.structure[j] == 2;
        st.structure[j] = if kept_city[j] { t.structure[j] } else if kept_rule[j] && village_or_ruin { 0 } else { board.structure[j] };
        st.road[j] = false;
        if !kept_city[j] {
            st.terrain[j] = board.terrain[j];
            st.resource[j] = board.resource[j];
            st.capital[j] = 0;
            st.structure_level[j] = 0;
            st.structure_turn[j] = 0;
        }
        if !kept_rule[j] {
            st.owner[j] = 0;
            st.city[j] = -1;
        }
    }
    // Crop/metal on seen tiles the searcher's tech cannot see: redraw from the sample.
    let tech = &pl.tech[p];
    let crop_hidden = !(tech[6] || tech[8] || tech[9]);
    let metal_hidden = !(tech[20] || tech[23] || tech[24]);
    let hidden = |r: i32| (r == CROP && crop_hidden) || (r == METAL && metal_hidden);
    for j in (0..n).filter(|&j| seen[j]) {
        let shown = if hidden(t.resource[j]) { 0 } else { t.resource[j] };
        if shown == 0 && (crop_hidden || metal_hidden) {
            st.resource[j] = if board.terrain[j] == t.terrain[j] && hidden(board.resource[j]) { board.resource[j] } else { 0 };
        }
    }

    // Units: drop enemies the searcher cannot see (an invisible one next to its units is flagged by detect).
    for i in 0..n {
        let detected = (0..n).any(|j| d(i, j) == 1 && u.owner[j] == me) || c.owner[i] == me;
        if u.owner[i] == enemy && (fog[i] || (u.invisible[i] && !detected)) {
            s.units.clear(i);
        }
    }
    for i in (0..n).filter(|&i| moved[i]) {
        copy_city(&mut s, i, &empty, i);
    }

    // Relocate hidden enemy cities (capital first) to free sampled village sites in the fog.
    let mut free: Vec<bool> = (0..n).map(|j| board.structure[j] == 1 && fog[j] && !(0..n).any(|k| d(j, k) <= 1 && seen[k])).collect();
    let mut cap_site = vec![false; n];
    for &k in &board.capitals {
        cap_site[k] = true;
    }
    let mut order: Vec<usize> = (0..n).filter(|&i| moved[i]).collect();
    order.sort_by_key(|&i| (t.capital[i] <= 0, i));
    for i in order {
        let ok: Vec<bool> = (0..n).map(|j| free[j] && s.tiles.owner[j] == 0 && s.units.owner[j] == 0).collect();
        let prefer_caps = t.capital[i] > 0 && (0..n).any(|j| ok[j] && cap_site[j]);
        let pool: Vec<bool> = (0..n).map(|j| ok[j] && (!prefer_caps || cap_site[j])).collect();
        let noise: Vec<f32> = (0..n).map(|_| rng.uniform()).collect();
        if !pool.iter().any(|&x| x) {
            s.units.home.iter_mut().filter(|h| **h == i as i32 + 1).for_each(|h| *h = 0);
            continue;
        }
        let key = |k: usize| if pool[k] { noise[k] } else { -1. };
        let j = (0..n).fold(0, |b, k| if key(k) > key(b) { k } else { b });
        let area: Vec<bool> = (0..n).map(|x| x == j || (d(j, x) <= c.border[i] && fog[x] && s.tiles.owner[x] == 0)).collect();
        copy_city(&mut s, j, state, i);
        s.cities.territory[j * n..(j + 1) * n].copy_from_slice(&area);
        for x in (0..n).filter(|&x| area[x]) {
            s.tiles.owner[x] = enemy;
            s.tiles.city[x] = j as i32;
        }
        s.tiles.structure[j] = 1;
        s.tiles.terrain[j] = t.terrain[i];
        s.tiles.resource[j] = 0;
        s.tiles.capital[j] = t.capital[i];
        s.tiles.structure_level[j] = t.structure_level[i];
        s.units.home.iter_mut().filter(|h| **h == i as i32 + 1).for_each(|h| *h = j as i32 + 1);
        for x in 0..n {
            free[x] &= d(j, x) > 2;
        }
    }

    // Enemy economy and knowledge: symmetric priors from the searcher's own position.
    s.players.stars[q] = (pl.stars[p] - income(&s, me, &e.rules, g)).max(0);
    s.players.tech[q] = pl.tech[p];
    let score = s.players.score;
    s.tiles.explored[q] = seen;
    update_exploration(&mut s, enemy, g);
    s.players.score = score;
    s.seed = rng.next_u64() as u32;
    s.initial_seed = rng.next_u64() as u32;
    s
}

/// The world plan round `round` searches: a determinized sample, or the true state once samples keep failing.
pub fn world(e: &Engine, truth: &State, sample: bool, seed: u64, round: u64) -> State {
    if sample { determinize(truth, e, &mut Rng::new(mix(seed, round, WORLD))) } else { truth.clone() }
}

pub struct Lane {
    pub actions: Vec<usize>,
    pub score: f32,
    pub steps: usize,
}

/// One search lane from `root` (search._lane_kit's advance, depth 1, root='sequence'): the searcher's turn
/// (greedy for lane 0, else sampled at T=1), then the opponent's turn sampled at T=1, scored by the critic of the
/// searcher's own view at its next decision; a finished game scores its outcome, an exhausted budget the last
/// value from the searcher's side.
pub fn run_lane(e: &mut Engine, p: &mut Policy, root: &State, greedy: bool, rng: &mut Rng) -> Result<Lane, String> {
    let me = root.current_player;
    let mut s = root.clone();
    let (mut changes, mut steps, mut actions) = (0, 0, Vec::new());
    loop {
        let mover = s.current_player;
        let trunk = p.trunk(e, &s, mover);
        if changes == 2 {
            return Ok(Lane { actions, score: trunk.value, steps });
        }
        let in_root = changes == 0;
        let mask = e.legal(&s, true).to_vec();
        let draw = if in_root && greedy { Draw::Greedy } else { Draw::Sample(1.) };
        let choice = p.decide(&e.catalog, &trunk, &mask, draw, rng);
        if in_root {
            actions.push(choice.index);
        }
        let status = e.apply(&mut s, choice.index);
        if status != OK {
            return Err(format!("search lane: the engine rejected a policy action (status {status})"));
        }
        changes += (s.current_player != mover) as usize;
        steps += 1;
        if s.terminated || s.truncated {
            let outcome = if s.terminated { terminal_rewards(&s)[(me - 1) as usize] } else { 0. };
            return Ok(Lane { actions, score: outcome, steps });
        }
        if steps >= MAX_STEPS || (in_root && actions.len() >= MAX_TURN_ACTIONS) {
            return Ok(Lane { actions, score: if mover == me { trunk.value } else { -trunk.value }, steps });
        }
    }
}

/// The best lane: the highest score, ties to the lower lane (lane 0 is the greedy turn).
pub fn best(scores: &[f32]) -> usize {
    (0..scores.len()).fold(0, |b, i| if scores[i] > scores[b] { i } else { b })
}

/// build_guarded_replay: plays `actions` on `truth` and on the sampled `world` side by side; stops before a
/// sub-action the true game rejects and right after the first that leaves the mover's view of the two different.
/// Returns how many were applied to `truth`.
pub fn commit(e: &mut Engine, truth: &mut State, world: &State, actions: &[usize]) -> usize {
    let me = truth.current_player;
    let mut b = world.clone();
    let mut applied = 0;
    for &a in actions.iter().take(MAX_TURN_ACTIONS) {
        if e.apply(truth, a) != OK {
            break;
        }
        applied += 1;
        if e.apply(&mut b, a) != OK || packed(truth, me, &e.geom) != packed(&b, me, &e.geom) {
            break;
        }
    }
    applied
}

/// Commits one round's chosen turn: guarded against the sampled world, or its first sub-action when the round
/// searched the true state. Returns the applied catalog entries (empty: the sample was rejected).
pub fn commit_round(e: &mut Engine, truth: &mut State, world: &State, sample: bool, chosen: &[usize]) -> Result<Vec<usize>, String> {
    if sample {
        let applied = commit(e, truth, world, chosen);
        return Ok(chosen[..applied].to_vec());
    }
    if e.apply(truth, chosen[0]) != OK {
        return Err("search fallback produced an illegal action".into());
    }
    Ok(vec![chosen[0]])
}

/// TurnSearcher.play_turn on one core: plans until the mover changes or the game ends. Returns the catalog
/// entries to play in order and metrics. web/src/local/search.ts runs the same rounds with lanes on workers.
pub fn plan_turn(e: &mut Engine, p: &mut Policy, s: &State, lanes: usize, seed: u64) -> Result<Value, String> {
    let mover = s.current_player;
    let mut truth = s.clone();
    let (mut plan, mut rejected, mut plans, mut overrides, mut fallbacks) = (Vec::new(), 0, 0, 0, 0);
    let mut value = None;
    let mut round = 0u64;
    while truth.current_player == mover && !truth.terminated && !truth.truncated {
        round += 1;
        let sample = rejected <= MAX_REJECTED;
        let w = world(e, &truth, sample, seed, round);
        let mut results = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            results.push(run_lane(e, p, &w, lane == 0, &mut Rng::new(mix(seed, round, lane as u64)))?);
        }
        let b = best(&results.iter().map(|l| l.score).collect::<Vec<_>>());
        let applied = commit_round(e, &mut truth, &w, sample, &results[b].actions)?;
        if applied.is_empty() {
            rejected += 1;
            continue;
        }
        fallbacks += !sample as usize;
        rejected = 0;
        plans += 1;
        overrides += (b != 0) as usize;
        value.get_or_insert(results[b].score);
        plan.extend(applied.iter().map(|i| i.to_string()));
    }
    Ok(json!({"plan": plan, "plans": plans, "overrides": overrides, "fallbacks": fallbacks, "value": value}))
}
