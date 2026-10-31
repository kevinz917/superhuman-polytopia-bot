//! The Explorer walk of the ruins and city-level Explorer rewards (game version 123): the game's explorer steps with
//! a fog-distance search. 12 steps; each step scores an 11x11 window around the explorer, moves to the best
//! of its 8 neighbours (ties drawn from the step's hash) and explores radius 1 around it (2 on a mountain).

use crate::geom::Geom;
use crate::hash::hash_words;
use crate::state::State;

/// Explorer moves.
pub const MOVES: usize = 12;
/// Scan radius around the explorer (an 11x11 window).
const RADIUS: i32 = 5;
const WIDTH: i32 = 2 * RADIUS + 1;
const CELLS: usize = (WIDTH * WIDTH) as usize;

// lookup flags
const VALID: u8 = 1;
const CAN_MOVE_OVER: u8 = 2;
const CAN_END_MOVE_ON: u8 = 4;
const FOG: u8 = 8;
const CHECK_FROM_TO: u8 = 16;
const LIGHTHOUSE: u8 = 32;
const MOUNTAIN: u8 = 64;
const READ_ONLY: u8 = 128;

/// Candidate steps in the game's order, (dx, dy).
const STEPS: [(i32, i32); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];
/// Last queue index the fog-distance relaxation processes (the game caps it).
const QUEUE_LIMIT: usize = 0x1f2;

/// Unlocked movement for Bardur: field, forest (and the unused 6/7) always, water with Fishing,
/// ocean with Sailing, mountain with Climbing.
#[inline]
pub fn movable(tech: &[bool; 128], terrain: i32) -> bool {
    match terrain {
        1 => tech[10],
        2 => tech[13],
        4 => tech[20],
        3 | 5 | 6 | 7 => true,
        _ => false,
    }
}

/// Score from the surrounding fog: fog tiles in the (clipped) square of radius `r` around `index`, capped
/// at 4; a fogged lighthouse in it lowers the score. Radius 2 with neither gives 9999.
fn fog_score(look: &[u8; CELLS], index: usize, r: i32) -> i32 {
    let (cx, cy) = (index as i32 % WIDTH, index as i32 / WIDTH);
    let (x0, x1) = ((cx - r).max(0), (cx + r).min(WIDTH - 1));
    let (y0, y1) = ((cy - r).max(0), (cy + r).min(WIDTH - 1));
    let (mut count, mut light) = (0, false);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let f = look[(y * WIDTH + x) as usize];
            count += (f & FOG != 0) as i32;
            light |= f & (FOG | LIGHTHOUSE) == FOG | LIGHTHOUSE;
        }
    }
    if r == 2 && count == 0 && !light {
        return 9999;
    }
    let v = count.min(4);
    if light { 40 - 10 * v } else { 73 - 10 * v }
}

/// Map tile of window cell `k` around (fx, fy), if it is on the board.
#[inline]
fn tile_of(size: i32, fx: i32, fy: i32, k: usize) -> Option<usize> {
    let (x, y) = (fx - RADIUS + k as i32 % WIDTH, fy - RADIUS + k as i32 / WIDTH);
    (x >= 0 && x < size && y >= 0 && y < size).then(|| (y * size + x) as usize)
}

/// The explorer's next tile from `from`, or None (the walk stops). `path` is
/// every tile walked so far, the start included.
fn next_step(s: &State, explored: &[bool], from: usize, path: &[usize], seed: u32) -> Option<usize> {
    let size = s.size as i32;
    let (fx, fy) = ((from % s.size) as i32, (from / s.size) as i32);
    let cp = s.current_player;
    let tech = &s.players.tech[s.p()];
    // fill the lookup window
    let mut look = [0u8; CELLS];
    for (k, cell) in look.iter_mut().enumerate() {
        let Some(t) = tile_of(size, fx, fy, k) else { continue };
        let seen = explored[t];
        let structure = s.tiles.structure[t];
        let terrain = s.tiles.terrain[t];
        let (bridge, city) = (structure == 48, structure == 1);
        let enter = if bridge || city { city } else { movable(tech, terrain) };
        *cell = VALID
            | if bridge { CHECK_FROM_TO } else { 0 }
            | if enter { CAN_MOVE_OVER | CAN_END_MOVE_ON } else { 0 }
            | if seen { 0 } else { FOG }
            | if structure == 47 { LIGHTHOUSE } else { 0 }
            | if terrain == 4 && seen { MOUNTAIN } else { 0 };
    }
    // the fog-distance map (mountains checked)
    let mut dist = [0i32; CELLS];
    let mut queue: Vec<usize> = Vec::with_capacity(4 * CELLS);
    for k in 0..CELLS {
        dist[k] = if look[k] & CAN_MOVE_OVER != 0 { 1000 } else { 9999 };
    }
    for k in 0..CELLS {
        if look[k] & FOG != 0 {
            dist[k] = 0;
            queue.push(k);
        }
    }
    for k in 0..CELLS {
        if look[k] & (MOUNTAIN | FOG | CAN_MOVE_OVER) == MOUNTAIN | CAN_MOVE_OVER {
            let score = fog_score(&look, k, 2);
            if score != 9999 {
                dist[k] = score + 100;
                queue.push(k);
                look[k] |= READ_ONLY;
            }
        }
    }
    let mut at = 0usize;
    while at < queue.len() {
        let u = queue[at] as i32;
        let right = if u % WIDTH < WIDTH - 1 { 1 } else { 9999 };
        let left = if u % WIDTH > 0 { -1 } else { -9999 };
        let step = dist[u as usize] + 100;
        for index in [u + WIDTH, u - WIDTH, u + right, u + left, u + left + WIDTH, u + right + WIDTH, u + left - WIDTH,
                      u + right - WIDTH] {
            // update the neighbour and enqueue it
            if index < 0 || index as usize >= CELLS {
                continue;
            }
            let i = index as usize;
            if dist[i] <= step || look[i] & READ_ONLY != 0 {
                continue;
            }
            // a step between two tiles (bridges): passability for a scout
            let crosses = look[i] & CHECK_FROM_TO != 0
                && match (tile_of(size, fx, fy, i), tile_of(size, fx, fy, u as usize)) {
                    (Some(to), Some(fr)) => explored[to] && !crate::kernels::bridge_blocks(s, cp, fr, to),
                    _ => false,
                };
            if !crosses && look[i] & (VALID | CAN_MOVE_OVER) != VALID | CAN_MOVE_OVER {
                continue;
            }
            dist[i] = if step == 100 && look[i] & FOG == 0 { fog_score(&look, i, 1) + 100 } else { step };
            queue.push(i);
        }
        if at == QUEUE_LIMIT {
            break;
        }
        at += 1;
    }
    // backtracking penalty: 1, 2, 3, ... along the path, or 1, 1, 2, 2, ... once it has 9 tiles
    let long = path.len() > 8;
    let (mut penalty, mut count) = (1, 0u32);
    for &t in path {
        let (lx, ly) = ((t % s.size) as i32 - (fx - RADIUS), (t / s.size) as i32 - (fy - RADIUS));
        if (0..WIDTH).contains(&lx) && (0..WIDTH).contains(&ly) {
            dist[(ly * WIDTH + lx) as usize] += penalty;
        }
        penalty += if long { (count & 1) as i32 } else { 1 };
        count += long as u32;
    }
    let mut best = i32::MAX;
    let mut ties = [0usize; 8];
    let mut n = 0;
    for (j, &(dx, dy)) in STEPS.iter().enumerate() {
        let v = dist[((dy + RADIUS) * WIDTH + dx + RADIUS) as usize];
        if v < best {
            best = v;
            n = 0;
        }
        if v == best {
            ties[n] = j;
            n += 1;
        }
    }
    if best > 9998 {
        return None;
    }
    // the seeded hash of `seed`, modulo count
    let pick = if n == 1 { ties[0] } else { ties[(hash_words(s.initial_seed, &[seed]) % n as u32) as usize] };
    let (dx, dy) = STEPS[pick];
    let (x, y) = (fx + dx, fy + dy);
    (x >= 0 && x < size && y >= 0 && y < size).then(|| (y * size + x) as usize)
}

/// The tiles the current player newly explores by an Explorer reward on `source` (ruins or a city reward;
/// both seed the walk with the seeded hash of the reward tile's (x, y)). The game seed is
/// our `initial_seed`.
pub fn explorer(s: &State, source: usize, g: &Geom) -> Vec<bool> {
    let n = s.n;
    let initial = &s.tiles.explored[s.p()];
    let mut explored: Vec<bool> = initial.clone();
    let mut seed = hash_words(s.initial_seed, &[(source % s.size) as u32, (source / s.size) as u32]);
    let mut path = Vec::with_capacity(MOVES + 1);
    path.push(source);
    let mut from = source;
    for _ in 0..MOVES {
        let Some(target) = next_step(s, &explored, from, &path, seed) else { break };
        path.push(target);
        // exploring from a tile: radius 1, 2 from a mountain
        let radius = if s.tiles.terrain[target] == 4 { 2 } else { 1 };
        g.for_ball(target, radius, |t| explored[t] = true);
        // the next step's seed: the seeded hash of [seed]
        seed = hash_words(s.initial_seed, &[seed]);
        from = target;
    }
    (0..n).map(|t| explored[t] && !initial[t]).collect()
}
