//! Procedural Bardur Drylands layout and reset (gpu/mapgen.py, maps/state.py::initial_state). Two seats are the
//! reference generator bit for bit; four seats run the same steps with one capital per quadrant.

use crate::geom::Geom;
use crate::hash::DotNetRandom;
use crate::kernels::update_exploration;
use crate::state::State;

pub fn ruins(size: usize) -> Option<i32> {
    match size {
        11 => Some(4),
        14 => Some(5),
        18 => Some(9),
        _ => None,
    }
}

pub fn round_limit(size: usize) -> i32 {
    if size == 18 { 150 } else { 100 }
}

pub struct Layout {
    pub terrain: Vec<i32>,
    pub structure: Vec<i32>,
    pub resource: Vec<i32>,
    /// One capital tile per seat.
    pub capitals: Vec<usize>,
}

/// mapgen._choose: bounded draw over the count of set entries, then the draw-th set index (0 when none).
fn choose(rng: &mut DotNetRandom, mask: &[bool]) -> usize {
    let count = mask.iter().filter(|&&m| m).count() as i32;
    let draw = rng.bounded(count.max(1));
    let mut seen = 0;
    for (i, &m) in mask.iter().enumerate() {
        if m {
            if seen == draw {
                return i;
            }
            seen += 1;
        }
    }
    0
}

pub fn layout(seed: i32, g: &Geom) -> Layout {
    layout_players(seed, g, 2)
}

/// layout for `np` seats (2..=4): capitals in distinct quadrants, each later one drawn among the free quadrants at
/// least half as far from the first capital as the farthest free one.
pub fn layout_players(seed: i32, g: &Geom, np: usize) -> Layout {
    assert!((2..=4).contains(&np));
    let (size, n) = (g.size, g.n);
    let s = size as i32;
    let d = |i: usize, j: usize| g.d(i, j);
    let mut rng = DotNetRandom::new(seed);
    let mut available = [true; 4];
    let mut caps = vec![0usize; np];
    let quadsize = s / 2;
    let centers_x: Vec<i32> = (0..4).map(|c| (c % 2) * quadsize + quadsize / 2).collect();
    let centers_y: Vec<i32> = (0..4).map(|c| (c / 2) * quadsize + quadsize / 2).collect();
    for i in 0..np {
        let (c0x, c0y) = ((caps[0] % size) as i32, (caps[0] / size) as i32);
        let sq: Vec<i32> = (0..4).map(|q| (centers_x[q] - c0x).pow(2) + (centers_y[q] - c0y).pow(2)).collect();
        let maximum = (0..4).filter(|&q| available[q]).map(|q| sq[q]).max().unwrap_or(-1);
        let allowed: Vec<bool> = (0..4).map(|q| available[q] && (i == 0 || sq[q] >= maximum.div_euclid(2))).collect();
        let q = choose(&mut rng, &allowed) as i32;
        let sx = ((q % 2) * quadsize + 2).min(s - 3);
        let ex = (((q % 2) + 1) * quadsize - 2).max(sx + 1).min(s - 2);
        let sy = ((q / 2) * quadsize + 2).min(s - 3);
        let ey = (((q / 2) + 1) * quadsize - 2).max(sy + 1).min(s - 2);
        let cx = rng.bounded(ex - sx);
        let cy = rng.bounded(ey - sy);
        available[q as usize] = false;
        caps[i] = ((cy + sy) * s + cx + sx) as usize;
    }
    let mut land = vec![false; n];
    for &c in &caps {
        land[c] = true;
    }
    let target_land = ((n as f64) * 0.95) as usize;
    let mut attempt = 0i32;
    while land.iter().filter(|&&l| l).count() < target_land && attempt < (n as i32) * 10 {
        let at = rng.bounded(n as i32) as usize;
        land[at] = true;
        attempt += 1;
    }
    let mut villages = vec![false; n];
    for &c in &caps {
        villages[c] = true;
    }
    // Biome traversal: one active array per seat with swap removal, kept only for the RNG stream.
    let mut active = vec![vec![0usize; n]; np];
    for i in 0..np {
        active[i][0] = caps[i];
    }
    let mut counts = vec![1usize; np];
    let mut done = villages.clone();
    let mut changed = true;
    let mut candidates = vec![false; n];
    while changed {
        changed = false;
        for i in 0..np {
            if counts[i] > 0 {
                let index = rng.bounded(counts[i] as i32) as usize;
                let at = active[i][index];
                for t in 0..n {
                    candidates[t] = d(at, t) <= 1 && !done[t];
                }
                let any_preferred = (0..n).any(|t| candidates[t] && land[t]);
                if any_preferred {
                    for t in 0..n {
                        candidates[t] = candidates[t] && land[t];
                    }
                }
                if candidates.iter().any(|&c| c) {
                    let chosen = choose(&mut rng, &candidates);
                    active[i][counts[i]] = chosen;
                    counts[i] += 1;
                    done[chosen] = true;
                    changed = true;
                } else {
                    active[i][index] = active[i][counts[i] - 1];
                    counts[i] -= 1;
                }
            }
        }
    }
    let mut terrain = vec![0i32; n];
    for i in 0..n {
        let coast = !land[i]
            && [1usize, 3, 4, 6].iter().any(|&k| g.nb_valid[i * 8 + k] && land[g.nb[i * 8 + k] as usize]);
        terrain[i] = if land[i] { 3 } else if coast { 1 } else { 2 };
    }
    let forest_cut = 0.14f32 + 0.38f32 * 1.5f32;
    for i in 0..n {
        if land[i] && !villages[i] {
            let value = rng.unit_float();
            terrain[i] = if value < 0.14f32 { 4 } else if value < forest_cut { 5 } else { 3 };
        }
    }
    let edge: Vec<i32> = (0..n)
        .map(|i| {
            let (x, y) = ((i % size) as i32, (i / size) as i32);
            x.min(s - 1 - x).min(y.min(s - 1 - y))
        })
        .collect();
    let village_candidates = |villages: &[bool], terrain: &[i32], out: &mut [bool]| {
        for i in 0..n {
            out[i] = land[i] && !villages[i] && terrain[i] != 4 && edge[i] >= 2 && edge[i] != 3
                && !(0..n).any(|j| d(i, j) < 3 && villages[j]);
        }
    };
    loop {
        village_candidates(&villages, &terrain, &mut candidates);
        if !candidates.iter().any(|&c| c) {
            break;
        }
        let at = choose(&mut rng, &candidates);
        villages[at] = true;
        terrain[at] = 3;
    }
    let mut above: Vec<i32> = villages.iter().map(|&v| if v { 1 } else { 0 }).collect();
    for i in 0..np {
        for _ in 0..2 {
            for t in 0..n {
                candidates[t] = d(caps[i], t) <= 1 && t != caps[i] && above[t] == 0 && terrain[t] != 2;
            }
            if candidates.iter().any(|&c| c) {
                let at = choose(&mut rng, &candidates);
                terrain[at] = 5;
                above[at] = 101;
            }
        }
    }
    for city in 0..n {
        if !villages[city] {
            continue;
        }
        let mut count = (0..n).filter(|&t| d(city, t) <= 1 && above[t] == 101).count() as i32;
        for rad in 1..3i32 {
            for at in 0..n {
                let kind = terrain[at];
                if d(city, at) <= rad && above[at] == 0 && kind != 2 {
                    let value = rng.unit_float();
                    let probability: f32 = if kind == 3 {
                        if rad == 1 { 0.18 } else { 0.06 }
                    } else if kind == 5 {
                        if rad == 1 && count >= 3 { 0.0 } else if rad == 1 { 0.38 } else { 0.12 }
                    } else if kind == 4 {
                        if rad == 1 { 0.11 } else { 0.03 }
                    } else {
                        0.5
                    };
                    let resource = if kind == 3 { 106 } else if kind == 5 { 101 } else if kind == 4 { 105 } else { 103 };
                    let added = value < probability;
                    above[at] = if added { resource } else { 0 };
                    if added && kind == 5 && rad == 1 {
                        count += 1;
                    }
                }
            }
        }
    }
    let resource_value: Vec<i32> = (0..n)
        .map(|t| {
            let a = above[t];
            (if a == 101 || a == 103 || a == 106 { 2 } else if a == 105 { 3 } else { 0 }) + (terrain[t] == 5) as i32
        })
        .collect();
    let scores: Vec<i32> = (0..np).map(|i| (0..n).filter(|&t| d(caps[i], t) <= 2).map(|t| resource_value[t]).sum()).collect();
    let best = *scores.iter().max().unwrap();
    for i in 0..np {
        if scores[i] + 4 < best {
            let mut order: Vec<usize> = (0..n).filter(|&t| d(caps[i], t) <= 2).collect();
            let count = order.len();
            for _ in 0..count {
                let a = rng.bounded(count as i32) as usize;
                let b = rng.bounded(count as i32) as usize;
                order.swap(a, b);
            }
            let (mut at, mut score) = (0usize, scores[i]);
            while at < count && score + 4 < best {
                let idx = order[at];
                if above[idx] == 0 && terrain[idx] == 5 {
                    above[idx] = 101;
                    score += 2;
                }
                at += 1;
            }
        }
    }
    let ruin_count = ruins(size).expect("supported board size");
    let (mut placed, mut attempt) = (0i32, 0i32);
    while placed < ruin_count && attempt < 2000 {
        let at = rng.bounded(n as i32) as usize;
        let valid = above[at] == 0 && terrain[at] > 2 && !(0..n).any(|t| d(at, t) <= 1 && (above[t] == 2 || villages[t]));
        if valid {
            above[at] = 2;
            placed += 1;
        }
        attempt += 1;
    }
    let (mut placed, mut attempt) = (0i32, 0i32);
    while placed < (n / 25) as i32 && attempt < 1000 {
        let at = rng.bounded(n as i32) as usize;
        let valid = terrain[at] <= 2 && above[at] == 0
            && !(0..n).any(|t| d(at, t) <= 1 && (above[t] == 108 || above[t] == 47 || above[t] == 1));
        if valid {
            above[at] = 108;
            placed += 1;
        }
        attempt += 1;
    }
    for &c in &g.corners {
        above[c] = 47;
    }
    Layout {
        terrain,
        structure: above.iter().map(|&a| if a < 100 { a } else { 0 }).collect(),
        resource: above.iter().map(|&a| if a >= 100 { a - 100 } else { 0 }).collect(),
        capitals: caps,
    }
}

/// maps/state.py::initial_state, bitwise mapgen.reset after the layout step.
pub fn initial_state(seed: u32, terrain: &[i32], structure: &[i32], resource: &[i32], caps: &[usize], g: &Geom,
                     max_turns: i32) -> State {
    let (size, n) = (g.size, g.n);
    let np = caps.len();
    let mut s = State::empty_players(size, np);
    s.seed = seed;
    s.initial_seed = seed;
    s.reference_max_turns = max_turns;
    s.tiles.terrain.copy_from_slice(terrain);
    s.tiles.structure.copy_from_slice(structure);
    s.tiles.resource.copy_from_slice(resource);
    for t in 0..n {
        // owner = max over seats of (area ? seat+1 : 0)
        let mut owner = 0;
        for k in 0..np {
            if g.d(caps[k], t) <= 1 {
                owner = owner.max(k as i32 + 1);
            }
        }
        s.tiles.owner[t] = owner;
        s.tiles.city[t] = if owner > 0 { caps[(owner - 1) as usize] as i32 } else { -1 };
    }
    for k in 0..np {
        let c = caps[k];
        let owner = k as i32 + 1;
        s.tiles.capital[c] = owner;
        s.cities.owner[c] = owner;
        s.cities.level[c] = 1;
        s.cities.production[c] = 1;
        s.cities.border[c] = 1;
        for t in 0..n {
            s.cities.territory[c * n + t] = g.d(c, t) <= 1;
        }
        s.cities.order[c] = 1;
        s.units.owner[c] = owner;
        s.units.kind[c] = 2;
        s.units.health[c] = 10.0;
        s.units.home[c] = c as i32 + 1;
        s.units.prev_x[c] = (c % size) as i32;
        s.units.prev_y[c] = (c / size) as i32;
    }
    for k in 0..np {
        s.players.stars[k] = 5;
        s.players.tech[k][0] = true;
        s.players.tech[k][15] = true;
        s.players.score[k] = 390;
    }
    for k in 0..np {
        update_exploration(&mut s, k as i32 + 1, g);
    }
    s
}

pub fn reset(seed: i32, g: &Geom, max_turns: Option<i32>) -> State {
    reset_players(seed, g, max_turns, 2)
}

pub fn reset_players(seed: i32, g: &Geom, max_turns: Option<i32>, np: usize) -> State {
    let board = layout_players(seed, g, np);
    initial_state(seed as u32, &board.terrain, &board.structure, &board.resource, &board.capitals, g,
                  max_turns.unwrap_or(round_limit(g.size)))
}
