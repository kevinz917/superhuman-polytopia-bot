//! gpu/core.py: the atomic action dispatcher, end of turn, terminal rewards; gpu/actions.py::decode.

use crate::capture;
use crate::cov;
use crate::diplomacy;
use crate::economy;
use crate::geom::Geom;
use crate::infiltration;
use crate::kernels::{max_health, movement_targets, research_cost, research_mask, MoveScratch};
use crate::military;
use crate::prof;
use crate::pathfinding::{reveal_tiles, search, PathScratch};
use crate::rules::{Rules, INFILTRATE};
use crate::state::{State, MAXP};
use crate::structures;

pub const OK: i32 = 0;
pub const INVALID_ACTION: i32 = 1;
pub const UNSUPPORTED_RULE: i32 = 2;
pub const STALE_VERSION: i32 = 3;
pub const EPISODE_DONE: i32 = 4;

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub kind: i32,
    pub source: i32,
    pub target: i32,
    pub parameter: i32,
}

/// Catalog widths of the 17 mask families (gpu/actions.py::decode) for board tiles n.
pub fn family_widths(n: usize) -> [usize; 17] {
    [n * n, n * n, n * 128, n * 128, 128, n, n * 3, n * 9, n, n, n, n, n, n, 6, 1, 1]
}

/// Row widths used by rl/actions.py::flatten_choice (second dim of `shapes`).
pub fn family_row_widths() -> [usize; 17] {
    [0, 0, 128, 128, 128, 1, 3, 9, 1, 1, 1, 1, 1, 1, 6, 1, 1]
}

/// rl.actions.flatten_choice followed by gpu.actions.decode.
pub fn decode(s: &State, family: i32, source: i32, target: i32) -> Action {
    let n = s.n as i64;
    let widths = family_widths(s.n);
    let mut rows = family_row_widths();
    rows[0] = s.n;
    rows[1] = s.n;
    let mut offsets = [0i64; 18];
    for i in 0..17 {
        offsets[i + 1] = offsets[i] + widths[i] as i64;
    }
    let f = family.clamp(0, 16) as usize;
    let flat = offsets[f] + source as i64 * rows[f] as i64 + target as i64;
    let group = (1..=17).filter(|&g| flat >= offsets[g]).count().min(16);
    let at = flat - offsets[group];
    let other = 3 - s.current_player;
    let a = |kind: i32, source: i64, target: i64, parameter: i64| Action {
        kind,
        source: source as i32,
        target: target as i32,
        parameter: parameter as i32,
    };
    match group {
        0 => a(1, at / n, at % n, 0),
        1 => a(2, at / n, at % n, 0),
        2 => a(4, at / 128, -1, at % 128),
        3 => a(6, -1, at / 128, at % 128),
        4 => a(7, -1, -1, at),
        5 => a(5, -1, at, 0),
        6 => a(3, -1, at / 3, at % 3 + 1),
        7 => a(9, -1, at / 9, at % 9),
        8 => a(8, at, -1, 0),
        9 => a(3, at, -1, 7),
        10 => a(3, at, -1, 16),
        11 => a(3, at, -1, 8),
        12 => a(3, at, -1, 9),
        13 => a(3, -1, at, 4),
        14 => {
            let k = (at.clamp(0, 5)) as usize;
            let (capital, _) = diplomacy::opponent_capital(s);
            let targets = [-1i64, 0, 1, capital as i64, -1, -1];
            let abilities = [18i64, 19, 19, 20, 21, 22];
            a(3, other as i64, targets[k], abilities[k])
        }
        15 => a(10, -1, -1, 0),
        _ => a(11, -1, -1, 0),
    }
}

/// Meeting reward: the star gift for meeting `other` (and the version 123 "away stars" with two seats).
fn meet_gift(s: &State, other: usize) -> i32 {
    let score = s.players.score[other];
    let mut gift = 1 + 2 * if score > 0 { ((score - 1).div_euclid(1000) + 1).min(5) } else { 0 };
    // game version >= 123 (away stars): in a two-player game, meeting player 1 pays 2 more
    if s.np == 2 && other == 0 {
        gift += 2;
    }
    gift
}

/// Seat `p` meets seat `other`: the star gift and `known`, once.
pub fn meet_pair(s: &mut State, p: usize, other: usize) {
    if p == other || s.players.known[p][other] {
        return;
    }
    s.players.stars[p] += meet_gift(s, other);
    s.players.known[p][other] = true;
}

/// The mover meets every seat it can see: a visible unit or an owned city on an explored tile
/// (as exploring does, through its contact check).
fn meet(s: &mut State) {
    if !s.fog {
        return; // `new` below is false: nothing changes
    }
    let p = s.p();
    let n = s.n;
    // every other seat in order; with two seats exactly the reference's single opponent
    for other in (0..s.np).filter(|&q| q != p) {
        if s.players.known[p][other] {
            continue;
        }
        let o = other as i32 + 1;
        let sees = (0..n).any(|t| s.tiles.explored[p][t]
            && ((s.units.owner[t] == o && !s.units.invisible[t]) || (s.tiles.structure[t] == 1 && s.cities.owner[t] == o)));
        if sees {
            meet_pair(s, p, other);
        }
    }
}

fn research(s: &mut State, a: Action, r: &Rules) {
    let p = s.p();
    let k = a.parameter.clamp(0, 127) as usize;
    let cost = research_cost(s, r, k);
    s.players.stars[p] -= cost;
    s.players.score[p] = s.players.score[p].wrapping_add(r.tech_tier[k] * 100);
    s.players.tech[p][k] = true;
    s.players.tech_turn[p][k] = s.turn;
    meet(s);
}

fn recover(s: &mut State, src: usize, r: &Rules) {
    let cp = s.current_player;
    let territory_owner = s.tiles.owner[src];
    let friendly = territory_owner == cp || (territory_owner > 0 && s.players.peace[s.p()][s.pclip(territory_owner)]);
    let amount = if friendly { 4.0 } else { 2.0 };
    let mh = max_health(s, r, src);
    let u = &mut s.units;
    let hp = if u.poison[src] { u.health[src] } else { (u.health[src] + amount).min(mh) };
    u.health[src] = hp;
    u.poison[src] = false;
    u.moved[src] = true;
    u.attacked[src] = true;
    meet(s);
}

fn move_adjacent(s: &mut State, source: usize, target: usize, r: &Rules, g: &Geom, ps: &mut PathScratch) {
    let cp = s.current_player;
    if s.units.owner[target] != 0 && s.units.owner[target] != cp {
        // moving onto a hidden unit: the unit on the target is revealed (and its owner met); the
        // mover stays where it is and keeps its move
        cov::hit(33);
        s.units.invisible[target] = false;
        meet(s);
        return;
    }
    // moving: the unit goes along the game's route, which treats hidden units as free tiles (they
    // stay hidden), and explores from every tile of it
    let found = search(s, source, target, r, g, ps);
    let extra = if !found { Some(target) } else { None };
    reveal_tiles(s, source, 0, extra, r, ps);
    military::relocate(s, source, target, r, g, false, Some(&ps.reveal));
    // a move faces the unit along the path's last step
    if ps.path.len() > 1 {
        s.units.direction[target] = military::direction(s.size, ps.path[1], target);
    }
    // moving: ending on another player's city counts as an attack on them
    if s.tiles.structure[target] == 1 && State::foreign(s.cities.owner[target], cp) {
        diplomacy::hostile(s);
    }
    meet(s);
}

fn end_turn(s: &mut State, r: &Rules, g: &Geom) {
    let n = s.n;
    let owner = s.current_player;
    let old = s.p();
    for i in 0..n {
        let u = &s.units;
        let idle = u.owner[i] == owner && !u.moved[i] && !u.attacked[i];
        if idle {
            if !u.poison[i] {
                let to = s.tiles.owner[i];
                let friendly = to == owner || (to > 0 && s.players.peace[old][s.pclip(to)]);
                let amount = if friendly { 4.0 } else { 2.0 };
                let mh = max_health(s, r, i);
                s.units.health[i] = (s.units.health[i] + amount).min(mh);
            }
            s.units.poison[i] = false;
        }
    }
    // ending a turn: the riot of an infiltrated city ends at the end of its owner's turn
    for i in 0..n {
        if s.cities.riot[i] && s.cities.owner[i] == owner {
            s.cities.riot[i] = false;
        }
    }
    let p = &s.players;
    let (new_owner, turn, native) = if s.np == 2 {
        let dead = [p.killed_turn[0] > 0 || p.resigned_turn[0] > 0, p.killed_turn[1] > 0 || p.resigned_turn[1] > 0];
        let new_owner = if dead[(2 - owner) as usize] { owner } else { 3 - owner };
        (new_owner, s.turn + (new_owner == 1) as i32, dead[0] || dead[1])
    } else {
        // the next seat still in the game, in seat order; a round ends when the order wraps
        let np = s.np as i32;
        let new_owner = (1..=np).map(|k| (owner - 1 + k) % np + 1).find(|&o| s.alive((o - 1) as usize)).unwrap_or(owner);
        (new_owner, s.turn + (new_owner <= owner) as i32, s.alive_count() <= 1)
    };
    let pacifist = if p.attacked[old] { 0 } else { p.pacifist[old] + p.tech[old][21] as i32 };
    s.players.pacifist[old] = pacifist;
    s.players.attacked[old] = false;
    s.current_player = new_owner;
    s.turn = turn;
    let limit = turn > s.reference_max_turns;
    if limit || native {
        if limit && !native { cov::hit(31); }
        s.truncated = limit && !native;
        s.terminated = native;
        return;
    }
    let frozen: Vec<bool> = s.units.frozen.clone();
    diplomacy::start_turn(s, new_owner);
    if diplomacy::peace_pending(s) {
        // the turn start's diplomacy check: receiving a peace request meets its sender
        let (p, q) = (s.p(), (2 - new_owner) as usize);
        meet_pair(s, p, q);
    }
    // the turn start explores nothing: tiles are explored when units move, train, upgrade or convert and when
    // borders grow
    meet(s);
    structures::start_turn(s, new_owner);
    let mut income = structures::income(s, new_owner, r, g);
    if !(turn > 0 && income > 0) {
        income = 0;
    }
    let no = (new_owner - 1) as usize;
    s.players.stars[no] += income;
    economy::start_level_ups(s, new_owner);
    for i in 0..n {
        if s.units.owner[i] == new_owner {
            s.units.moved[i] = frozen[i];
            s.units.attacked[i] = frozen[i];
            if !frozen[i] {
                s.units.attacks_performed[i] = 0;
            }
            s.units.frozen[i] = false;
        }
    }
}

fn resign(s: &mut State) {
    let p = s.p();
    s.players.resigned_turn[p] = s.turn.max(1);
    if s.np == 2 {
        s.terminated = true;
        s.players.killer[p] = 3 - s.current_player;
    } else {
        // the seat leaves (its pieces stay as obstacles; the turn order skips it); the game ends at one seat left
        leave(s, p);
        s.terminated = s.alive_count() <= 1;
    }
}

/// Record that seat `k` has just left a multi-seat game (killed or resigned): its place in the leaving order.
pub fn leave(s: &mut State, k: usize) {
    if s.np > 2 && s.players.out_order[k] == 0 {
        s.players.out_order[k] = s.players.out_order.iter().filter(|&&o| o > 0).count() as i32 + 1;
    }
}

/// Reward of finishing place `place` (0 = first) among `np` seats: evenly spaced from +1 down to -1, zero-sum.
#[inline]
pub fn place_reward(place: usize, np: usize) -> f32 {
    1.0 - 2.0 * place as f32 / (np - 1) as f32
}

/// Seat-indexed final rewards (rows >= np are 0). Two seats: the reference's win +1 / loss -1, 0 unless terminated.
/// More seats: rank rewards (place_reward), set at termination and at truncation. Seats that left take the
/// places from last upward in the order they left; the seats still in share the top places equally (a seat
/// still in at the round cap ties with every other one still in).
pub fn terminal_rewards(s: &State) -> [f32; MAXP] {
    let p = &s.players;
    let mut out = [0f32; MAXP];
    if s.np == 2 {
        let dead = [p.killed_turn[0] > 0 || p.resigned_turn[0] > 0, p.killed_turn[1] > 0 || p.resigned_turn[1] > 0];
        if s.terminated {
            for k in 0..2 {
                let winner = !dead[k] && dead[1 - k];
                out[k] = if winner { 1.0 } else if dead[k] { -1.0 } else { 0.0 };
            }
        }
        return out;
    }
    if !(s.terminated || s.truncated) {
        return out;
    }
    let np = s.np;
    let alive = s.alive_count();
    let shared = (0..alive).map(|k| place_reward(k, np)).sum::<f32>() / alive.max(1) as f32;
    for k in 0..np {
        out[k] = if s.alive(k) { shared } else { place_reward(np - p.out_order[k].clamp(1, np as i32) as usize, np) };
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Research, Recover, Move, End, Attack, Summon, Harvest, Forest, Promote, Disband, Reward, Build, Destroy, Cap,
    Heal, Diplomatic, Resign, Unsupported,
}

fn classify(a: Action) -> Kind {
    match (a.kind, a.parameter) {
        (7, _) => Kind::Research,
        (3, 7) => Kind::Recover,
        (1, _) => Kind::Move,
        (10, _) => Kind::End,
        (2, _) => Kind::Attack,
        (4, _) => Kind::Summon,
        (5, _) => Kind::Harvest,
        (3, 1..=3) => Kind::Forest,
        (3, 16) => Kind::Promote,
        (3, 8) => Kind::Disband,
        (9, _) => Kind::Reward,
        (6, _) => Kind::Build,
        (3, 4) => Kind::Destroy,
        (8, _) => Kind::Cap,
        (3, 9) => Kind::Heal,
        (3, 18..=22) => Kind::Diplomatic,
        (11, _) => Kind::Resign,
        _ => Kind::Unsupported,
    }
}

/// Scratch buffers reused across steps.
pub struct Scratch {
    pub row: Vec<bool>,
    pub mv: MoveScratch,
    pub build: structures::BuildCtx,
    pub summon: military::SummonCtx,
    pub path: PathScratch,
}

impl Scratch {
    pub fn new(n: usize) -> Scratch {
        Scratch { row: vec![false; n * 9], mv: MoveScratch::new(n), build: structures::BuildCtx::new(n),
                  summon: military::SummonCtx::new(n), path: PathScratch::new(n) }
    }
}

/// core.step: apply one atomic action; rejection leaves the state untouched. Returns the status code.
pub fn step(s: &mut State, a: Action, expected_version: i32, r: &Rules, g: &Geom, max_actions: i32, sc: &mut Scratch) -> i32 {
    step_with(s, a, expected_version, r, g, max_actions, sc, None)
}

/// step, given optionally the packed legal masks (masks::Packed layout) computed for exactly this state: Move,
/// Build and Summon are then validated by one mask lookup. The status is the same either way: where a mask entry
/// and step's own predicate can differ (a pending city reward, the kind-37 build, an unsupported unit kind, a
/// core_supported=False state) the action is unsupported regardless of validity.
#[allow(clippy::too_many_arguments)]
pub fn step_with(s: &mut State, a: Action, expected_version: i32, r: &Rules, g: &Geom, max_actions: i32, sc: &mut Scratch,
                 legal: Option<&[u8]>) -> i32 {
    let n = s.n as i32;
    let legal = if s.core_supported { legal } else { None };
    let o = crate::masks::family_offsets(s.n);
    let bit = |b: &[u8], i: usize| b[i >> 3] >> (7 - (i & 7)) & 1 == 1;
    let src = a.source.clamp(0, n - 1) as usize;
    let dst = a.target.clamp(0, n - 1) as usize;
    let param = a.parameter.clamp(0, 127) as usize;
    let source_valid = a.source >= 0 && a.source < n;
    let target_valid = a.target >= 0 && a.target < n;
    let param_valid = a.parameter >= 0 && a.parameter < 128;
    let cp = s.current_player;
    let kind = classify(a);
    let own = source_valid && s.units.owner[src] == cp;
    let idle = own && !s.units.moved[src] && !s.units.attacked[src];
    let pending = s.pending_reward();
    let reward = kind == Kind::Reward;
    let reward_implemented = matches!(a.parameter, 1..=8);
    // pending command triggers: a peace request (answered first) or a city reward block every other command
    let peace = diplomacy::peace_pending(s);
    let mut supported = s.core_supported && kind != Kind::Unsupported
        && (if peace { kind == Kind::Diplomatic || kind == Kind::Resign } else { !pending || reward || kind == Kind::Resign })
        && (!reward || reward_implemented);
    if kind == Kind::Summon {
        supported &= r.unit_supported[param];
    }
    if kind == Kind::Build {
        supported &= param != 37;
    }
    let valid = prof::time(prof::VALIDATE + kind as usize, || match kind {
        Kind::End | Kind::Resign => true,
        Kind::Research => {
            let mut m = [false; 128];
            research_mask(s, r, &mut m);
            param_valid && m[param]
        }
        Kind::Recover => idle && s.units.health[src] < max_health(s, r, src),
        Kind::Move => source_valid && target_valid && match legal {
            Some(b) => bit(b, o[0] + src * s.n + dst),
            None => {
                movement_targets(s, src, r, g, &mut sc.mv, &mut sc.row);
                sc.row[dst]
            }
        },
        Kind::Attack => {
            military::attack_mask(s, src, r, g, &mut sc.row);
            source_valid && target_valid && sc.row[dst]
        }
        Kind::Summon => source_valid && param_valid && match legal {
            Some(b) => bit(b, o[2] + src * 128 + param),
            None => {
                sc.summon.prepare(s, r);
                military::summon_row(s, &sc.summon, src) >> param & 1 == 1
            }
        },
        Kind::Harvest => {
            economy::harvest_mask(s, &mut sc.row);
            target_valid && sc.row[dst]
        }
        Kind::Forest => {
            economy::forest_masks(s, &mut sc.row);
            target_valid && sc.row[dst * 3 + (a.parameter - 1).clamp(0, 2) as usize]
        }
        Kind::Promote => own && military::promotable(s, r, src),
        Kind::Disband => idle && s.players.tech[s.p()][2],
        Kind::Build => target_valid && param_valid && match legal {
            Some(b) => bit(b, o[3] + dst * 128 + param),
            None => structures::build_legal(s, r, g, &mut sc.build, dst, param),
        },
        Kind::Destroy => {
            structures::destroy_mask(s, &mut sc.row);
            target_valid && sc.row[dst]
        }
        Kind::Heal => {
            idle && s.units.kind[src] == 10
                && (0..s.n).any(|i| g.d(src, i) == 1 && s.units.owner[i] == cp && s.units.health[i] < max_health(s, r, i))
        }
        Kind::Cap => {
            capture::capture_mask(s, &mut sc.row);
            source_valid && sc.row[src]
        }
        Kind::Diplomatic => diplomacy::legal(s, a.source, a.target, a.parameter),
        Kind::Reward => {
            economy::reward_mask(s, &mut sc.row);
            target_valid && a.parameter >= 0 && a.parameter < 9 && sc.row[dst * 9 + a.parameter.clamp(0, 8) as usize]
        }
        Kind::Unsupported => false,
    });
    let mut status = if !supported { UNSUPPORTED_RULE } else if valid { OK } else { INVALID_ACTION };
    if expected_version != s.state_version {
        status = STALE_VERSION;
    }
    if s.terminated || s.truncated {
        status = EPISODE_DONE;
    }
    if status != OK {
        return status;
    }
    if !s.monuments_settled {
        prof::time(prof::STEP_PRE, || structures::remember_monuments(s, r, g));
    }
    prof::time(prof::EXECUTE + kind as usize, || match kind {
        Kind::Research => { cov::hit(0); research(s, a, r) }
        Kind::Recover => { cov::hit(1); recover(s, src, r) }
        Kind::Move => { cov::hit(2); move_adjacent(s, src, dst, r, g, &mut sc.path) }
        Kind::End => { cov::hit(3); end_turn(s, r, g) }
        Kind::Attack => {
            let uk = s.units.kind[src] as usize;
            if r.skills[uk][INFILTRATE] {
                cov::hit(6);
                infiltration::execute(s, src, dst, r, g);
            } else if uk == 10 {
                cov::hit(5);
                military::convert(s, src, dst, r, g);
            } else {
                cov::hit(4);
                military::attack(s, src, dst, r, g);
            }
        }
        Kind::Summon => {
            if s.units.kind[src] == 43 {
                cov::hit(8);
                military::upgrade(s, src, param, r, g);
            } else {
                cov::hit(7);
                military::summon(s, src, param, r, g, true);
            }
        }
        Kind::Harvest => { cov::hit(9); economy::harvest(s, dst) }
        Kind::Forest => { cov::hit(10); economy::forest(s, dst, a.parameter) }
        Kind::Promote => { cov::hit(11); military::promote(s, src, r) }
        Kind::Disband => { cov::hit(12); military::disband(s, src, r) }
        Kind::Reward => { cov::hit(13); if a.parameter == 6 { cov::hit(38); } if a.parameter == 4 { cov::hit(39); } economy::reward(s, dst, a.parameter, r, g) }
        Kind::Build => { cov::hit(14); structures::build(s, dst, param, r, g) }
        Kind::Destroy => { cov::hit(15); structures::destroy(s, dst, r, g) }
        Kind::Cap => {
            let st = s.tiles.structure[src];
            if st == 1 {
                cov::hit(16);
                if s.tiles.owner[src] > 0 { cov::hit(17); }
                capture::city(s, src, r, g);
            } else if st == 2 {
                capture::ruin(s, src, r, g);
            } else {
                cov::hit(23);
                capture::starfish(s, src);
            }
        }
        Kind::Heal => { cov::hit(24); military::heal_others(s, src, r, g) }
        Kind::Diplomatic => { cov::hit(25); diplomacy::execute(s, a.target, a.parameter, r, g) }
        Kind::Resign => { cov::hit(26); resign(s) }
        Kind::Unsupported => unreachable!(),
    });
    prof::time(prof::STEP_POST, || {
        if a.kind == 2 {
            diplomacy::hostile(s);
        }
        meet(s);
        diplomacy::capital_vision(s);
        structures::remember_monuments(s, r, g);
    });
    s.monuments_settled = true;
    let count = s.actions + 1;
    s.state_version += 1;
    s.actions = count;
    if count >= max_actions && !s.terminated && !s.truncated { cov::hit(32); }
    s.truncated = (s.truncated || count >= max_actions) && !s.terminated;
    OK
}
